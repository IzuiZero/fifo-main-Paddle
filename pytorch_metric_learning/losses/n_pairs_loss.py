import paddle

from ..distances import DotProductSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class NPairsLoss(BaseMetricLossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(name="num_pairs", is_stat=True)

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(
            indices_tuple, labels
        )
        self.num_pairs = len(anchor_idx)
        if self.num_pairs == 0:
            return self.zero_losses()
        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        targets = c_f.to_device(paddle.arange(self.num_pairs), embeddings)
        sim_mat = self.distance(anchors, positives)
        if not self.distance.is_inverted:
            sim_mat = -sim_mat

        loss_dict = {
            "loss": {
                "losses": paddle.nn.functional.cross_entropy(sim_mat, targets, reduction='none'),
                "indices": anchor_idx.numpy(),
                "reduction_type": "element",
            }
        }
        return loss_dict

    def get_default_distance(self):
        return DotProductSimilarity()
