import numpy as np
import paddle
import paddle.nn.functional as F

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class AngularLoss(BaseMetricLossFunction):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        alpha: The angle (as described in the paper), specified in degrees.
    """

    def __init__(self, alpha=40, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.alpha = paddle.to_tensor(np.radians(alpha), dtype='float32')
        self.add_to_recordable_attributes(list_of_names=["alpha"], is_stat=False)
        self.add_to_recordable_attributes(list_of_names=["average_angle"], is_stat=True)

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchors, positives, keep_mask, anchor_idx = self.get_pairs(
            embeddings, labels, indices_tuple
        )
        if anchors is None:
            return self.zero_losses()

        sq_tan_alpha = paddle.tan(self.alpha) ** 2
        ap_dot = paddle.sum(anchors * positives, axis=1, keepdim=True)
        ap_matmul_embeddings = paddle.matmul(
            (anchors + positives), embeddings.unsqueeze(2)
        )
        ap_matmul_embeddings = ap_matmul_embeddings.squeeze(2).t()

        final_form = (4 * sq_tan_alpha * ap_matmul_embeddings) - (
            2 * (1 + sq_tan_alpha) * ap_dot
        )
        losses = lmu.logsumexp(final_form, keep_mask=keep_mask, add_one=True)
        return {
            "loss": {
                "losses": losses,
                "indices": anchor_idx,
                "reduction_type": "element",
            }
        }

    def get_pairs(self, embeddings, labels, indices_tuple):
        a1, p, a2, _ = lmu.convert_to_pairs(indices_tuple, labels)
        if len(a1) == 0 or len(a2) == 0:
            return [None] * 4
        anchors = self.distance.normalize(embeddings[a1])
        positives = self.distance.normalize(embeddings[p])
        keep_mask = labels[a1].unsqueeze(1) != labels.unsqueeze(0)
        self.set_stats(anchors, positives, embeddings, keep_mask)
        return anchors, positives, keep_mask, a1

    def set_stats(self, anchors, positives, embeddings, keep_mask):
        if self.collect_stats:
            with paddle.no_grad():
                centers = (anchors + positives) / 2
                ap_dist = self.distance.pairwise_distance(anchors, positives)
                nc_dist = self.distance.get_norm(
                    centers - embeddings.unsqueeze(1), axis=2
                ).t()
                angles = paddle.atan(ap_dist.unsqueeze(1) / (2 * nc_dist))
                average_angle = paddle.sum(angles[keep_mask]) / paddle.sum(keep_mask)
                self.average_angle = np.degrees(average_angle.item())
