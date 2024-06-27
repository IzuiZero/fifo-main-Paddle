import paddle
import paddle.nn.functional as F

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class NCALoss(BaseMetricLossFunction):
    def __init__(self, softmax_scale=1, **kwargs):
        super().__init__(**kwargs)
        self.softmax_scale = softmax_scale
        self.add_to_recordable_attributes(
            list_of_names=["softmax_scale"], is_stat=False
        )

    # https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    def compute_loss(self, embeddings, labels, indices_tuple):
        if len(embeddings) <= 1:
            return self.zero_losses()
        return self.nca_computation(
            embeddings, embeddings, labels, labels, indices_tuple
        )

    def nca_computation(
        self, query, reference, query_labels, reference_labels, indices_tuple
    ):
        dtype = query.dtype
        miner_weights = lmu.convert_to_weights(indices_tuple, query_labels, dtype=dtype)
        mat = self.distance(query, reference)
        if not self.distance.is_inverted:
            mat = -mat
        if paddle.equal(query, reference):
            mat = paddle.fill_diagonal(mat, float('-inf'))

        same_labels = paddle.to_tensor(query_labels.unsqueeze(1) == reference_labels.unsqueeze(0))
        exp = F.softmax(self.softmax_scale * mat, axis=1)
        exp = paddle.sum(exp * same_labels, axis=1)
        non_zero = exp != 0
        loss = -paddle.log(exp[non_zero]) * miner_weights[non_zero]
        return {
            "loss": {
                "losses": loss,
                "indices": c_f.paddle_arange_from_size(query.shape[0])[non_zero],
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return LpDistance(power=2)
