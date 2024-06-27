import paddle

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseTupleMiner


class MultiSimilarityMiner(BaseTupleMiner):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_to_recordable_attributes(name="epsilon", is_stat=False)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels.numpy(), ref_labels.numpy())

        if len(a1) == 0 or len(a2) == 0:
            empty = paddle.to_tensor([], dtype='int64')
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat.numpy()
        mat_pos_sorting = mat.clone()

        dtype = mat.dtype
        pos_ignore = (
            c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        )
        neg_ignore = (
            c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)
        )

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        if embeddings.numpy() is ref_emb.numpy():
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = paddle.sort(mat_pos_sorting, axis=1)
        neg_sorted, neg_sorted_idx = paddle.sort(mat_neg_sorting, axis=1)

        if self.distance.is_inverted:
            hard_pos_idx = paddle.nonzero(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1), as_tuple=False
            )
            hard_neg_idx = paddle.nonzero(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1), as_tuple=False
            )
        else:
            hard_pos_idx = paddle.nonzero(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1), as_tuple=False
            )
            hard_neg_idx = paddle.nonzero(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1), as_tuple=False
            )

        a1 = hard_pos_idx[:, 0]
        p = pos_sorted_idx[a1, hard_pos_idx[:, 1]]
        a2 = hard_neg_idx[:, 0]
        n = neg_sorted_idx[a2, hard_neg_idx[:, 1]]

        return a1, p, a2, n

    def get_default_distance(self):
        return CosineSimilarity()
