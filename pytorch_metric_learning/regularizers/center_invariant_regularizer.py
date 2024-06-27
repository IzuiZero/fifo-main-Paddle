import paddle
import paddle.nn.functional as F

from ..distances import LpDistance
from ..utils import common_functions as c_f
from .base_regularizer import BaseRegularizer


class CenterInvariantRegularizer(BaseRegularizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, LpDistance, power=1, normalize_embeddings=False)

    def compute_loss(self, weights):
        squared_weight_norms = self.distance.get_norm(weights) ** 2
        deviations_from_mean = squared_weight_norms - paddle.mean(squared_weight_norms)
        return {
            "loss": {
                "losses": (deviations_from_mean ** 2) / 4,
                "indices": paddle.arange(weights.shape[0], dtype='int64'),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return LpDistance(power=1, normalize_embeddings=False)
