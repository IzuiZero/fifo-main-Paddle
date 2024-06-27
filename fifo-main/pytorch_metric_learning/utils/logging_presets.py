import os
import shutil
import sqlite3
import logging

import paddle

from . import common_functions as c_f

LOGGER = logging.getLogger(__name__)

class HookContainer:
    def __init__(
        self,
        record_keeper,
        record_group_name_prefix=None,
        primary_metric="mean_average_precision_at_r",
        validation_split_name="val",
        save_models=True,
    ):
        self.record_keeper = record_keeper
        self.record_group_name_prefix = record_group_name_prefix
        self.saveable_trainer_objects = [
            "models",
            "optimizers",
            "lr_schedulers",
            "loss_funcs",
            "mining_funcs",
        ]
        self.primary_metric = primary_metric
        self.validation_split_name = validation_split_name
        self.do_save_models = save_models

    ############################################
    ############################################
    ##################  HOOKS  #################
    ############################################
    ############################################

    ### Define the end_of_iteration hook. This will be executed at the end of every iteration. ###
    def end_of_iteration_hook(self, trainer):
        record_these = [
            [
                trainer.loss_tracker.losses,
                {"input_group_name_for_non_objects": "loss_histories"},
            ],
            [
                trainer.loss_tracker.loss_weights,
                {"input_group_name_for_non_objects": "loss_weights"},
            ],
            [trainer.loss_funcs, {"recursive_types": [paddle.nn.Layer]}],
            [trainer.mining_funcs, {}],
            [trainer.models, {}],
            [trainer.optimizers, {"custom_attr_func": self.optimizer_custom_attr_func}],
        ]
        for record, kwargs in record_these:
            self.record_keeper.update_records(
                record, trainer.get_global_iteration(), **kwargs
            )

    # This hook will be passed into the trainer and will be executed at the end of every epoch.
    def end_of_epoch_hook(
        self,
        tester,
        dataset_dict,
        model_folder,
        test_interval=1,
        patience=None,
        splits_to_eval=None,
        test_collate_fn=None,
    ):
        if self.primary_metric not in tester.accuracy_calculator.get_curr_metrics():
            raise ValueError(
                "HookContainer `primary_metric` must be one of: {}".format(
                    tester.accuracy_calculator.get_curr_metrics()
                )
            )
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        def actual_hook(trainer):
            continue_training = True
            if trainer.epoch % test_interval == 0:
                best_epoch = self.save_models_and_eval(
                    trainer,
                    dataset_dict,
                    model_folder,
                    test_interval,
                    tester,
                    splits_to_eval,
                    test_collate_fn,
                )
                continue_training = self.patience_remaining(
                    trainer.epoch, best_epoch, patience
                )
            return continue_training

        return actual_hook

    def end_of_testing_hook(self, tester):
        for split_name, accuracies in tester.all_accuracies.items():
            epoch = accuracies["epoch"]
            self.record_keeper.update_records(
                accuracies,
                epoch,
                input_group_name_for_non_objects=self.record_group_name(
                    tester, split_name
                ),
            )
            _, _, best_epoch, best_accuracy = self.is_new_best_accuracy(
                tester, split_name, epoch
            )
            best = {"best_epoch": best_epoch, "best_accuracy": best_accuracy}
            self.record_keeper.update_records(
                best,
                epoch,
                input_group_name_for_non_objects=self.record_group_name(
                    tester, split_name
                ),
            )

    ############################################
    ############################################
    ######### MODEL LOADING AND SAVING #########
    ############################################
    ############################################

    def load_latest_saved_models(self, trainer, model_folder, device=None, best=False):
        if device is None:
            device = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        resume_epoch, model_suffix = c_f.latest_version(
            model_folder, "trunk_*.pdparams", best=best
        )
        if resume_epoch > 0:
            for obj_dict in [
                getattr(trainer, x, {}) for x in self.saveable_trainer_objects
            ]:
                c_f.load_dict_of_models(
                    obj_dict, model_suffix, model_folder, device, log_if_successful=True
                )
        return resume_epoch + 1

    def save_models(self, trainer, model_folder, curr_suffix, prev_suffix=None):
        if self.do_save_models:
            for obj_dict in [
                getattr(trainer, x, {}) for x in self.saveable_trainer_objects
            ]:
                c_f.save_dict_of_models(obj_dict, curr_suffix, model_folder)
                if prev_suffix is not None:
                    c_f.delete_dict_of_models(obj_dict, prev_suffix, model_folder)

    def save_models_and_eval(
        self,
        trainer,
        dataset_dict,
        model_folder,
        test_interval,
        tester,
        splits_to_eval=None,
        collate_fn=None,
    ):
        epoch = trainer.epoch
        tester.test(
            dataset_dict,
            epoch,
            trainer.models["trunk"],
            trainer.models["embedder"],
            splits_to_eval,
            collate_fn,
        )
        prev_best_epoch, _ = self.get_best_epoch_and_accuracy(
            tester, self.validation_split_name
        )
        (
            is_new_best,
            curr_accuracy,
            best_epoch,
            best_accuracy,
        ) = self.is_new_best_accuracy(tester, self.validation_split_name, epoch)
        self.record_keeper.save_records()
        trainer.step_lr_plateau_schedulers(curr_accuracy)
        self.save_models(
            trainer, model_folder, epoch, epoch - test_interval
        )  # save latest model
        if is_new_best:
            c_f.LOGGER.info("New best accuracy! {}".format(curr_accuracy))
            curr_suffix = "best%d" % best_epoch
            prev_suffix = (
                "best%d" % prev_best_epoch if prev_best_epoch is not None else None
            )
            self.save_models(
                trainer, model_folder, curr_suffix, prev_suffix
            )  # save best model
        return best_epoch

    def is_new_best_accuracy(self, tester, split_name, epoch):
        curr_accuracy = self.get_curr_primary_metric(tester, split_name)
        best_epoch, best_accuracy = self.get_best_epoch_and_accuracy(tester, split_name)
        is_new_best = False
        if (curr_accuracy > best_accuracy) or (best_epoch is None):
            best_epoch, best_accuracy = epoch, curr_accuracy
            is_new_best = True
        return is_new_best, curr_accuracy, best_epoch, best_accuracy

    ############################################
    ############################################
    ##### BEST EPOCH AND ACCURACY TRACKING #####
    ############################################
    ############################################

    def get_loss_history(self, loss_names=()):
        columns = "*" if len(loss_names) == 0 else ", ".join(loss_names)
        table_name = "loss_histories"
        if not self.record_keeper.table_exists(table_name):
            return {}
        output = self.record_keeper.query(
            "SELECT {} FROM {}".format(columns, table_name), return_dict=True
        )
        output.pop("id", None)
        return output

    def get_accuracy_history(
        self, tester, split_name, return_all_metrics=False, metrics=()
    ):
        table_name = self.record_group_name(tester, split_name)

        if not self.record_keeper.table_exists(table_name):
            return {}

        def get_accuracies(keys):
            keys = "*" if return_all_metrics else ", ".join(keys)
            query = "SELECT {} FROM {}".format(keys, table_name)
            return self.record_keeper.query(query, return_dict=True)

        return get_accuracies(tester.accuracy_calculator.get_curr_metrics())

    def get_accuracies_of_epoch(self, tester, split_name, epoch):
        table_name = self.record_group_name(tester, split_name)
        return self.record_keeper.query(
            f"SELECT * FROM {table_name} WHERE epoch=?", (epoch,)
        )

    def get_accuracies_of_best_epoch(
        self, tester, split_name, select_all=True, ignore_epoch=(-1,)
    ):
        table_name = self.record_group_name(tester, split_name)
        if not self.record_keeper.table_exists(table_name):
            return [], None

        def get_accuracies(key):
            columns = "*" if select_all else "epoch, %s" % key
            params = ", ".join(["?"] * len(ignore_epoch))
            query = """SELECT {0} FROM {1} WHERE {2}=
                        (SELECT max({2}) FROM {1} WHERE epoch NOT IN ({3}))
                        AND epoch NOT IN ({3})""".format(
                columns, table_name, key, params
            )
            output = self.record_keeper.query(query, ignore_epoch + ignore_epoch)
            return output, key

        return self.try_primary_metric(tester, get_accuracies)

    def get_best_epoch_and_accuracy(self, tester, split_name, ignore_epoch=(-1,)):
        accuracies, key = self.get_accuracies_of_best_epoch(
            tester, split_name, select_all=False, ignore_epoch=ignore_epoch
        )
        if len(accuracies) > 0:
            return accuracies[0]["epoch"], accuracies[0][key]
        return None, 0

    def patience_remaining(self, epoch, best_epoch, patience):
        if patience is not None and best_epoch is not None:
            if epoch - best_epoch > patience:
                c_f.LOGGER.info("Validation accuracy has plateaued. Exiting.")
                return False
        return True

    def run_tester_separately(
        self,
        tester,
        dataset_dict,
        epoch,
        trunk,
        embedder,
        splits_to_eval=None,
        collate_fn=None,
        skip_eval_if_already_done=True,
    ):
        if skip_eval_if_already_done:
            splits_to_eval = self.get_splits_to_eval(
                tester, dataset_dict, epoch, splits_to_eval
            )
            if len(splits_to_eval) == 0:
                c_f.LOGGER.info("Already evaluated")
                return False
        tester.test(dataset_dict, epoch, trunk, embedder, splits_to_eval, collate_fn)
        return True

    def get_splits_to_eval(self, tester, dataset_dict, epoch, input_splits_to_eval):
        input_splits_to_eval = (
            list(dataset_dict.keys())
            if input_splits_to_eval is None
            else input_splits_to_eval
        )
        splits_to_eval = []
        for split in input_splits_to_eval:
            if len(self.get_accuracies_of_epoch(tester, split, epoch)) == 0:
                splits_to_eval.append(split)
        return splits_to_eval

    def base_record_group_name(self, tester):
        base_record_group_name = (
            "%s_" % self.record_group_name_prefix
            if self.record_group_name_prefix
            else ""
        )
        base_record_group_name += tester.description_suffixes("accuracies")
        return base_record_group_name

    def record_group_name(self, tester, split_name):
        base_record_group_name = self.base_record_group_name(tester)
        query_set_label = split_name.upper()
        reference_sets_label = "_and_".join(
            [name.upper() for name in sorted(tester.reference_split_names[split_name])]
        )
        if reference_sets_label == query_set_label:
            reference_sets_label = "self"
        return f"{base_record_group_name}_{query_set_label}_vs_{reference_sets_label}"

    def optimizer_custom_attr_func(self, optimizer):
        return {"lr": optimizer.param_groups[0]["lr"]}


class EmptyContainer:
    def end_of_epoch_hook(self, *args):
        return None

    end_of_iteration_hook = None
    end_of_testing_hook = None


def get_record_keeper(
    csv_folder,
    tensorboard_folder=None,
    global_db_path=None,
    experiment_name=None,
    is_new_experiment=True,
    save_figures=False,
    save_lists=False,
):
    try:
        import record_keeper as record_keeper_package
        from paddle.fluid.writer import SummaryWriter

        record_writer = record_keeper_package.RecordWriter(
            folder=csv_folder,
            global_db_path=global_db_path,
            experiment_name=experiment_name,
            is_new_experiment=is_new_experiment,
            save_lists=save_lists,
        )
        tensorboard_writer = (
            SummaryWriter(logdir=tensorboard_folder)
            if tensorboard_folder is not None
            else None
        )
        record_keeper = record_keeper_package.RecordKeeper(
            tensorboard_writer=tensorboard_writer,
            record_writer=record_writer,
            attributes_to_search_for=c_f.list_of_recordable_attributes_list_names(),
            save_figures=save_figures,
        )
        return record_keeper, record_writer, tensorboard_writer
    except ModuleNotFoundError as e:
        c_f.LOGGER.warning(e)
        c_f.LOGGER.warning("There won't be any logging or model saving.")
        c_f.LOGGER.warning("To fix this, pip install record-keeper tensorboard")
        return None, None, None


def get_hook_container(record_keeper, **kwargs):
    if record_keeper:
        return HookContainer(record_keeper, **kwargs)
    else:
        c_f.LOGGER.warning("No record_keeper, so no preset hooks are being returned.")
        return EmptyContainer()
