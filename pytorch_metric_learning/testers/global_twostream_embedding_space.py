import tqdm

from ..utils import common_functions as c_f
from .global_embedding_space import GlobalEmbeddingSpaceTester


class GlobalTwoStreamEmbeddingSpaceTester(GlobalEmbeddingSpaceTester):
    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        s, e = 0, 0
        all_anchors, all_posnegs, labels = None, None, None
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            anchors, posnegs, label = self.data_and_label_getter(data)
            label = c_f.process_label(
                label, self.label_hierarchy_level, self.label_mapper
            )
            a = self.get_embeddings_for_eval(trunk_model, embedder_model, anchors)
            pns = self.get_embeddings_for_eval(trunk_model, embedder_model, posnegs)
            if label.ndim == 1:
                label = label.unsqueeze(1)
            if i == 0:
                labels = paddle.zeros(len(dataloader.dataset), label.shape[1])
                all_anchors = paddle.zeros(len(dataloader.dataset), pns.shape[1])
                all_posnegs = paddle.zeros(len(dataloader.dataset), pns.shape[1])

            e = s + pns.shape[0]
            all_anchors[s:e] = a
            all_posnegs[s:e] = pns
            labels[s:e] = label
            s = e
        return all_anchors, all_posnegs, labels

    def get_all_embeddings(self, dataset, trunk_model, embedder_model, collate_fn):
        dataloader = c_f.get_eval_dataloader(
            dataset, self.batch_size, self.dataloader_num_workers, collate_fn
        )
        anchor_embeddings, posneg_embeddings, labels = self.compute_all_embeddings(
            dataloader, trunk_model, embedder_model
        )
        anchor_embeddings, posneg_embeddings = (
            self.maybe_normalize(anchor_embeddings),
            self.maybe_normalize(posneg_embeddings),
        )
        return (
            paddle.concat([anchor_embeddings, posneg_embeddings], axis=0),
            paddle.concat([labels, labels], axis=0),
        )

    def set_reference_and_query(
        self, embeddings_and_labels, query_split_name, reference_split_names
    ):
        assert (
            query_split_name == reference_split_names[0]
            and len(reference_split_names) == 1
        ), "{} does not support different reference and query splits".format(
            self.__class__.__name__
        )
        embeddings, labels = embeddings_and_labels[query_split_name]
        half = int(embeddings.shape[0] / 2)
        anchors_embeddings = embeddings[:half]
        posneg_embeddings = embeddings[half:]
        query_labels = labels[:half]
        return anchors_embeddings, query_labels, posneg_embeddings, query_labels

    def embeddings_come_from_same_source(self, query_split_name, reference_split_names):
        return False
