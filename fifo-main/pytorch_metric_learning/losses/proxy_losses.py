import paddle

from ..utils import common_functions as c_f
from .mixins import WeightRegularizerMixin
from .nca_loss import NCALoss


class ProxyNCALoss(WeightRegularizerMixin, NCALoss):
    def __init__(self, num_classes, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.proxies = paddle.create_parameter(
            shape=[num_classes, embedding_size],
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform(),
        )
        self.proxy_labels = paddle.arange(num_classes)
        self.add_to_recordable_attributes(list_of_names=["num_classes"], is_stat=False)

    def cast_types(self, dtype, device):
        self.proxies.set_value(c_f.to_device(self.proxies.numpy(), device=device, dtype=dtype))

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        loss_dict = self.nca_computation(
            embeddings,
            self.proxies,
            labels,
            c_f.to_device(self.proxy_labels, device=device, dtype=labels.dtype),
            indices_tuple,
        )
        self.add_weight_regularization_to_loss_dict(loss_dict, self.proxies.numpy())
        return loss_dict
