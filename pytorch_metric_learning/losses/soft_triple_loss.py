import paddle
import paddle.nn.functional as F

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction
from .mixins import WeightRegularizerMixin

###### modified from https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py ######
###### Original code is Copyright@Alibaba Group ######
###### ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling" ######
class SoftTripleLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(
        self,
        num_classes,
        embedding_size,
        centers_per_class=10,
        la=20,
        gamma=0.1,
        margin=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert self.distance.is_inverted
        self.la = la
        self.gamma = 1.0 / gamma
        self.margin = margin
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.fc = paddle.nn.Parameter(
            paddle.Tensor(embedding_size, num_classes * centers_per_class)
        )
        self.weight_init_func(self.fc)
        self.add_to_recordable_attributes(
            list_of_names=[
                "la",
                "gamma",
                "margin",
                "centers_per_class",
                "num_classes",
                "embedding_size",
            ],
            is_stat=False,
        )

    def cast_types(self, dtype, device):
        self.fc.data = c_f.to_device(self.fc.data, device=device, dtype=dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        sim_to_centers = self.distance(embeddings, paddle.transpose(self.fc, [1, 0]))
        sim_to_centers = sim_to_centers.reshape(
            [-1, self.num_classes, self.centers_per_class]
        )
        prob = F.softmax(sim_to_centers * self.gamma, axis=2)
        sim_to_classes = paddle.sum(prob * sim_to_centers, axis=2)
        margin = paddle.zeros(
            sim_to_classes.shape, dtype=dtype, device=embeddings.device
        )
        margin[paddle.arange(0, margin.shape[0]), labels] = self.margin
        loss = F.cross_entropy(
            self.la * (sim_to_classes - margin), labels, reduction="none"
        )
        loss = loss * miner_weights
        loss_dict = {
            "loss": {
                "losses": loss,
                "indices": c_f.paddle_arange_from_size(embeddings.shape[0]),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, paddle.transpose(self.fc, [1, 0]))
        return loss_dict

    def get_default_distance(self):
        return CosineSimilarity()

    def get_default_weight_init_func(self):
        return c_f.PaddleInitWrapper(paddle.nn.initializer.KaimingUniform(a=math.sqrt(5)))
