import paddle
import paddle.nn.functional as F

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class FastAPLoss(BaseMetricLossFunction):
    def __init__(self, num_bins=10, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, LpDistance, normalize_embeddings=True, p=2)
        self.num_bins = int(num_bins)
        self.num_edges = self.num_bins + 1
        self.add_to_recordable_attributes(list_of_names=["num_bins"], is_stat=False)

    """
    Adapted from https://github.com/kunhe/FastAP-metric-learning
    """

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.place
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        N = labels.shape[0]
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = paddle.zeros((N, N), dtype=dtype)
        I_neg = paddle.zeros((N, N), dtype=dtype)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1
        N_pos = paddle.sum(I_pos, axis=1)
        safe_N = N_pos > 0
        if paddle.sum(safe_N) == 0:
            return self.zero_losses()
        dist_mat = self.distance(embeddings)

        histogram_max = 2 ** self.distance.power
        histogram_delta = histogram_max / self.num_bins
        mid_points = paddle.linspace(
            0.0, histogram_max, num=self.num_edges, dtype=dtype
        ).reshape([-1, 1, 1])
        pulse = F.relu(
            1 - paddle.abs(dist_mat - mid_points) / histogram_delta
        )
        pos_hist = paddle.transpose(paddle.sum(pulse * I_pos, axis=2), perm=[1, 0])
        neg_hist = paddle.transpose(paddle.sum(pulse * I_neg, axis=2), perm=[1, 0])

        total_pos_hist = paddle.cumsum(pos_hist, axis=1)
        total_hist = paddle.cumsum(pos_hist + neg_hist, axis=1)

        h_pos_product = pos_hist * total_pos_hist
        safe_H = (h_pos_product > 0) & (total_hist > 0)
        if paddle.sum(safe_H) > 0:
            FastAP = paddle.zeros_like(pos_hist)
            FastAP = paddle.where(safe_H, h_pos_product / total_hist, FastAP)
            FastAP = paddle.sum(FastAP, axis=1)
            FastAP = paddle.where(safe_N, FastAP / N_pos, paddle.zeros_like(FastAP))
            FastAP = (1 - FastAP) * miner_weights[safe_N]
            return {
                "loss": {
                    "losses": FastAP,
                    "indices": paddle.where(safe_N)[0],
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()

    def get_default_distance(self):
        return LpDistance(power=2)
