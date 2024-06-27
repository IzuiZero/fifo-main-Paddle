import paddle

from ..utils import loss_and_miner_utils as lmu  # Assuming the import path is correct
from .base_distance import BaseDistance  # Assuming BaseDistance is imported correctly

class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.place
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == paddle.float16:  # cdist doesn't work for float16
            rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, axis=0)
            output = paddle.zeros(rows.shape, dtype=dtype, place=device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            return paddle.dist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb, ref_emb):
        return paddle.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)
