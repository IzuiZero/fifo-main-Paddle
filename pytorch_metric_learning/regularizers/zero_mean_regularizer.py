import paddle

from ..utils import common_functions as c_f
from .base_regularizer import BaseRegularizer


class ZeroMeanRegularizer(BaseRegularizer):
    def compute_loss(self, embeddings):
        return {
            "loss": {
                "losses": paddle.abs(paddle.sum(embeddings, axis=1)),
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
