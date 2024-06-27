import paddle
from ..distances import CosineSimilarity
from ..reducers import DivisorReducer
from ..utils import common_functions as c_f
from .base_regularizer import BaseRegularizer

class SparseCentersRegularizer(BaseRegularizer):
    def __init__(self, num_classes, centers_per_class, **kwargs):
        super().__init__(**kwargs)
        assert centers_per_class > 1
        c_f.assert_distance_type(self, CosineSimilarity)
        self.set_class_masks(num_classes, centers_per_class)
        self.add_to_recordable_attributes(
            list_of_names=["num_classes", "centers_per_class"], is_stat=False
        )
        self.add_to_recordable_attributes(
            list_of_names=["same_class_center_sim", "diff_class_center_sim"],
            is_stat=True,
        )

    def compute_loss(self, weights):
        center_similarities = self.distance(weights)
        small_val = c_f.small_val(weights.numpy().dtype)
        center_similarities_masked = paddle.clip(
            2.0 * center_similarities[self.same_class_mask], max=2
        )
        divisor = 2 * paddle.sum(self.same_class_mask)
        reg = paddle.sqrt(2.0 + small_val - center_similarities_masked)
        self.set_stats(center_similarities)
        return {
            "loss": {
                "losses": reg,
                "indices": c_f.paddle_arange_from_size(reg),
                "reduction_type": "element",
                "divisor": divisor,
            }
        }

    def set_class_masks(self, num_classes, centers_per_class):
        total_num_centers = num_classes * centers_per_class
        self.diff_class_mask = paddle.ones(
            total_num_centers, total_num_centers, dtype='bool'
        )
        self.same_class_mask = paddle.zeros(
            total_num_centers, total_num_centers, dtype='bool'
        )
        for i in range(num_classes):
            s, e = i * centers_per_class, (i + 1) * centers_per_class
            curr_block = paddle.ones((centers_per_class, centers_per_class))
            curr_block = paddle.triu(curr_block, diagonal=1)
            self.same_class_mask[s:e, s:e] = curr_block
            self.diff_class_mask[s:e, s:e] = 0

    def set_stats(self, center_similarities):
        if self.collect_stats:
            self.same_class_center_sim = paddle.mean(
                center_similarities[self.same_class_mask]
            ).item()
            self.diff_class_center_sim = paddle.mean(
                center_similarities[self.diff_class_mask]
            ).item()

    def get_default_distance(self):
        return CosineSimilarity()

    def get_default_reducer(self):
        return DivisorReducer()
