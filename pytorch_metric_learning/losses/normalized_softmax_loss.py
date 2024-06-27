import paddle

from ..distances import DotProductSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction
from .mixins import WeightRegularizerMixin


class NormalizedSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(self, num_classes, embedding_size, temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.W = paddle.nn.Parameter(paddle.Tensor(embedding_size, num_classes))
        self.weight_init_func(self.W)
        self.cross_entropy = paddle.nn.CrossEntropyLoss(reduction="none")
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "num_classes", "temperature"],
            is_stat=False,
        )

    def cast_types(self, dtype, device):
        self.W.data = c_f.to_device(self.W.data, device=device, dtype=dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        normalized_W = self.distance.normalize(self.W, dim=0)
        exponent = self.distance(embeddings, normalized_W.t()) / self.temperature
        if not self.distance.is_inverted:
            exponent = -exponent
        unweighted_loss = self.cross_entropy(exponent, labels)
        miner_weighted_loss = unweighted_loss * miner_weights
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.paddle_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, paddle.transpose(self.W))
        return loss_dict

    def get_default_distance(self):
        return DotProductSimilarity()
