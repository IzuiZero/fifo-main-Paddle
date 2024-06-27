import copy
import paddle

from ..distances import LpDistance
from ..losses import TripletMarginLoss
from ..miners import EmbeddingsAlreadyPackagedAsTriplets
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .train_with_classifier import TrainWithClassifier


class NegativeLpDistance(LpDistance):
    def compute_mat(self, query_emb, ref_emb):
        return super().compute_mat(query_emb, ref_emb) * -1

    def pairwise_distance(self, query_emb, ref_emb):
        return super().pairwise_distance(query_emb, ref_emb) * -1


class DeepAdversarialMetricLearning(TrainWithClassifier):
    def __init__(
        self,
        metric_alone_epochs=0,
        g_alone_epochs=0,
        g_triplets_per_anchor=100,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.original_loss_weights = copy.deepcopy(self.loss_weights)
        self.metric_alone_epochs = metric_alone_epochs
        self.g_alone_epochs = g_alone_epochs
        assert isinstance(self.loss_funcs["g_adv_loss"], TripletMarginLoss)
        self.loss_funcs["g_adv_loss"].margin *= -1
        self.loss_funcs["g_adv_loss"].distance = NegativeLpDistance()
        self.g_triplets_per_anchor = g_triplets_per_anchor

    def custom_setup(self):
        synth_packaged_as_triplets = EmbeddingsAlreadyPackagedAsTriplets()
        self.mining_funcs["synth_packaged_as_triplets"] = synth_packaged_as_triplets
        self.loss_names += ["g_hard_loss", "g_reg_loss"]

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        penultimate_embeddings = self.get_trunk_output(data)

        if self.do_metric:
            authentic_final_embeddings = self.get_final_embeddings(
                penultimate_embeddings
            )
            indices_tuple = self.maybe_mine_embeddings(
                authentic_final_embeddings, labels
            )
            self.losses["metric_loss"] = self.loss_funcs["metric_loss"](
                authentic_final_embeddings, labels, indices_tuple
            )
            logits = self.maybe_get_logits(authentic_final_embeddings)
            self.losses["classifier_loss"] = self.maybe_get_classifier_loss(
                logits, labels
            )

        if self.do_adv:
            self.calculate_synth_loss(penultimate_embeddings, labels)

    def update_loss_weights(self):
        self.do_metric_alone = self.epoch <= self.metric_alone_epochs
        self.do_adv_alone = (
            self.metric_alone_epochs
            < self.epoch
            <= self.metric_alone_epochs + self.g_alone_epochs
        )
        self.do_both = not self.do_adv_alone and not self.do_metric_alone
        self.do_adv = self.do_adv_alone or self.do_both
        self.do_metric = self.do_metric_alone or self.do_both

        non_zero_weight_list = []
        if self.do_adv:
            non_zero_weight_list += ["g_hard_loss", "g_reg_loss", "g_adv_loss"]
        if self.do_metric:
            non_zero_weight_list += ["metric_loss", "classifier_loss"]
        if self.do_both:
            non_zero_weight_list += ["synth_loss"]

        for k in self.loss_weights.keys():
            if k in non_zero_weight_list:
                self.loss_weights[k] = self.original_loss_weights[k]
            else:
                self.loss_weights[k] = 0

        self.maybe_exclude_networks_from_gradient()

    def maybe_exclude_networks_from_gradient(self):
        self.set_to_train()
        if self.do_adv_alone:
            no_grad_list = ["trunk", "classifier"]
        elif self.do_metric_alone:
            no_grad_list = ["generator"]
        else:
            no_grad_list = []
        for k in self.models.keys():
            if k in no_grad_list:
                c_f.set_requires_grad(self.models[k], requires_grad=False)
                self.models[k].eval()
            else:
                c_f.set_requires_grad(self.models[k], requires_grad=True)

    def step_optimizers(self):
        step_list = []
        if self.do_metric:
            step_list += [
                "trunk_optimizer",
                "embedder_optimizer",
                "classifier_optimizer",
            ]
        if self.do_adv:
            step_list += ["generator_optimizer"]
        for k in self.optimizers.keys():
            if k in step_list:
                self.optimizers[k].step()

    def calculate_synth_loss(self, penultimate_embeddings, labels):
        a_indices, p_indices, n_indices = lmu.convert_to_triplets(
            None, labels, t_per_anchor=self.g_triplets_per_anchor
        )
        real_anchors = paddle.gather(penultimate_embeddings, paddle.to_tensor(a_indices))
        real_positives = paddle.gather(penultimate_embeddings, paddle.to_tensor(p_indices))
        real_negatives = paddle.gather(penultimate_embeddings, paddle.to_tensor(n_indices))
        penultimate_embeddings_cat = paddle.concat(
            [real_anchors, real_positives, real_negatives], axis=1
        )
        synthetic_negatives = self.models["generator"](
            c_f.to_device(penultimate_embeddings_cat, device=self.data_device)
        )
        penultimate_embeddings_with_negative_synth = c_f.unslice_by_n(
            [real_anchors, real_positives, synthetic_negatives]
        )
        final_embeddings = self.get_final_embeddings(
            penultimate_embeddings_with_negative_synth
        )

        labels = paddle.to_tensor(
            [
                val
                for tup in zip(
                    *[labels[a_indices], labels[p_indices], labels[n_indices]]
                )
                for val in tup
            ]
        )

        indices_tuple = self.mining_funcs["synth_packaged_as_triplets"](
            final_embeddings, labels
        )

        if self.do_both:
            self.losses["synth_loss"] = self.loss_funcs["synth_loss"](
                final_embeddings, labels, indices_tuple
            )

        self.losses["g_adv_loss"] = self.loss_funcs["g_adv_loss"](
            final_embeddings, labels, indices_tuple
        )
        self.losses["g_hard_loss"] = paddle.nn.functional.mse_loss(
            paddle.nn.functional.normalize(synthetic_negatives, p=2, axis=1),
            paddle.nn.functional.normalize(real_anchors, p=2, axis=1),
        )
        self.losses["g_reg_loss"] = paddle.nn.functional.mse_loss(
            paddle.nn.functional.normalize(synthetic_negatives, p=2, axis=1),
            paddle.nn.functional.normalize(real_negatives, p=2, axis=1),
        )

    def modify_schema(self):
        super().modify_schema()
        self.schema["models"].keys += ["generator"]
        self.schema["models"].essential += ["generator"]
        self.schema["loss_funcs"].keys += ["synth_loss", "g_adv_loss"]
        self.schema["loss_funcs"].essential += ["synth_loss", "g_adv_loss"]
        self.schema["mining_funcs"].keys += ["synth_packaged_as_triplets"]
