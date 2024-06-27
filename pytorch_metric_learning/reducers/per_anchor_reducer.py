import paddle
import paddle.nn.functional as F

from ..utils import common_functions as c_f
from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer


def aggregation_func(x, num_per_row):
    zero_denom = num_per_row == 0
    x = paddle.sum(x, axis=1) / num_per_row
    x = paddle.where(zero_denom, paddle.zeros_like(x), x)
    return x


class PerAnchorReducer(BaseReducer):
    def __init__(self, reducer=None, aggregation_func=aggregation_func, **kwargs):
        super().__init__(**kwargs)
        self.reducer = reducer if reducer is not None else MeanReducer()
        self.aggregation_func = aggregation_func

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        loss_dict = {
            "loss": {
                "losses": losses,
                "indices": loss_indices,
                "reduction_type": "element",
            }
        }
        return self.reducer(loss_dict, embeddings, labels)

    def tuple_reduction_helper(self, losses, loss_indices, embeddings, labels):
        batch_size = embeddings.shape[0]
        device, dtype = losses.place, losses.dtype
        new_array = paddle.full((batch_size, batch_size), c_f.pos_inf(dtype), dtype=dtype)
        
        anchors, others = loss_indices
        new_array[anchors, others] = losses
        pos_inf_mask = new_array == c_f.pos_inf(dtype)
        num_inf = paddle.sum(pos_inf_mask, axis=1)

        new_array = paddle.where(pos_inf_mask, paddle.zeros_like(new_array), new_array)
        num_per_row = batch_size - num_inf
        output = self.aggregation_func(new_array, num_per_row)

        loss_dict = {
            "loss": {
                "losses": output,
                "indices": paddle.arange(embeddings.shape[0], dtype='int64'),
                "reduction_type": "element",
            }
        }
        return self.reducer(loss_dict, embeddings, labels)

    def pos_pair_reduction(self, *args, **kwargs):
        return self.tuple_reduction_helper(*args, **kwargs)

    def neg_pair_reduction(self, *args, **kwargs):
        return self.tuple_reduction_helper(*args, **kwargs)

    def triplet_reduction(self, *args, **kwargs):
        raise NotImplementedError("Triplet reduction not supported")
