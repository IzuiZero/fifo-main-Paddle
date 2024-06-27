import paddle

from .base_distance import BaseDistance  # Assuming BaseDistance is imported correctly

class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        return paddle.matmul(query_emb, paddle.transpose(ref_emb, perm=[1, 0]))

    def pairwise_distance(self, query_emb, ref_emb):
        return paddle.sum(query_emb * ref_emb, axis=1)
