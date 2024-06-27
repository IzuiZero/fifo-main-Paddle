import numpy as np
import paddle

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .generic_pair_loss import GenericPairLoss


class TupletMarginLoss(GenericPairLoss):
    def __init__(self, margin=5.73, scale=64, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.margin = np.radians(margin)
        self.scale = scale
        self.add_to_recordable_attributes(
            list_of_names=["margin", "scale"], is_stat=False
        )
        self.add_to_recordable_attributes(
            list_of_names=["avg_pos_angle", "avg_neg_angle"], is_stat=True
        )

    # pos_pairs and neg_pairs already represent cos(theta)
    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            pos_angles = paddle.acos(pos_pairs)
            self.set_stats(pos_angles, neg_pairs)
            pos_pairs = paddle.cos(pos_angles - self.margin)
            pos_pairs = paddle.unsqueeze(pos_pairs, axis=1)
            neg_pairs = paddle.tile(neg_pairs, [pos_pairs.shape[0], 1])
            inside_exp = self.scale * (neg_pairs - pos_pairs)
            keep_mask = paddle.unsqueeze(a2, axis=0) == paddle.unsqueeze(a1, axis=1)
            loss = lmu.logsumexp(inside_exp, keep_mask=keep_mask, add_one=True, axis=1)
            return {
                "loss": {
                    "losses": loss,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

    def get_default_distance(self):
        return CosineSimilarity()

    def set_stats(self, pos_angles, neg_pairs):
        if self.collect_stats:
            neg_angles = paddle.acos(neg_pairs)
            self.avg_pos_angle = np.degrees(paddle.mean(pos_angles).numpy())
            self.avg_neg_angle = np.degrees(paddle.mean(neg_angles).numpy())
