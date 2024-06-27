import paddle
from .base_reducer import BaseReducer

class ThresholdReducer(BaseReducer):
    def __init__(self, low=None, high=None, **kwargs):
        super().__init__(**kwargs)
        assert (low is not None) or (high is not None), "At least one of low or high must be specified"
        self.low = low
        self.high = high
        if self.low is not None:
            self.add_to_recordable_attributes(list_of_names=["low"], is_stat=False)
        if self.high is not None:
            self.add_to_recordable_attributes(list_of_names=["high"], is_stat=False)

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "elements")

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "pos_pairs")

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "neg_pairs")

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "triplets")

    def element_reduction_helper(self, losses, embeddings, attr_name):
        low_condition = losses > self.low if self.low is not None else paddle.ones_like(losses, dtype='bool')
        high_condition = losses < self.high if self.high is not None else paddle.ones_like(losses, dtype='bool')
        threshold_condition = paddle.logical_and(low_condition, high_condition)
        num_past_filter = paddle.sum(threshold_condition)
        if num_past_filter >= 1:
            loss = paddle.mean(losses[threshold_condition])
        else:
            loss = self.zero_loss(embeddings)
        self.set_stats(low_condition, high_condition, num_past_filter, attr_name)
        return loss

    def set_stats(self, low_condition, high_condition, num_past_filter, attr_name):
        if self.collect_stats:
            curr_attr_name = "{}_past_filter".format(attr_name)
            self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
            setattr(self, curr_attr_name, num_past_filter.item())
            if self.low is not None:
                curr_attr_name = "{}_above_low".format(attr_name)
                self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                setattr(self, curr_attr_name, paddle.sum(low_condition).item())
            if self.high is not None:
                curr_attr_name = "{}_below_high".format(attr_name)
                self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                setattr(self, curr_attr_name, paddle.sum(high_condition).item())
