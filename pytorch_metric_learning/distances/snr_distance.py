import paddle
import paddle.nn as nn

from .base_distance import BaseDistance


# Signal to Noise Ratio
class SNRDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        anchor_variances = paddle.var(query_emb, axis=1)
        pairwise_diffs = paddle.unsqueeze(query_emb, axis=1) - ref_emb
        pairwise_variances = paddle.var(pairwise_diffs, axis=2)
        return pairwise_variances / (paddle.unsqueeze(anchor_variances, axis=1))

    def pairwise_distance(self, query_emb, ref_emb):
        query_var = paddle.var(query_emb, axis=1)
        query_ref_var = paddle.var(query_emb - ref_emb, axis=1)
        return query_ref_var / query_var
