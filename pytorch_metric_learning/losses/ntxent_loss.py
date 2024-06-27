import paddle

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from .generic_pair_loss import GenericPairLoss


class NTXentLoss(GenericPairLoss):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = paddle.to_tensor(float('-inf'), dtype=dtype)

            max_val = paddle.max(
                pos_pairs, paddle.max(neg_pairs, axis=1, keepdim=True)[0]
            ).detach()
            numerator = paddle.exp(pos_pairs - max_val).squeeze(1)
            denominator = paddle.sum(paddle.exp(neg_pairs - max_val), axis=1) + numerator
            log_exp = paddle.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

    def get_default_distance(self):
        return CosineSimilarity()
