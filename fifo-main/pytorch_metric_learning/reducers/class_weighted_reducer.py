import paddle
import numpy as np
from ..utils import common_functions as c_f
from .base_reducer import BaseReducer

class ClassWeightedReducer(BaseReducer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices, labels)

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices[0], labels)

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices[0], labels)

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, loss_indices[0], labels)

    def element_reduction_helper(self, losses, indices, labels):
        self.weights = c_f.to_device(self.weights, losses, dtype=losses.dtype)
        return paddle.mean(losses * self.weights[paddle.to_tensor(labels)[indices]])
