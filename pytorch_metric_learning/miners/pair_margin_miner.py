import paddle
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseTupleMiner

class PairMarginMiner(BaseTupleMiner):
    """
    Returns positive pairs that have distance greater than a margin and negative
    pairs that have distance less than a margin
    """

    def __init__(self, pos_margin=0.2, neg_margin=0.8, **kwargs):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.add_to_recordable_attributes(
            list_of_names=["pos_margin", "neg_margin"], is_stat=False
        )
        self.add_to_recordable_attributes(
            list_of_names=["pos_pair_dist", "neg_pair_dist"], is_stat=True
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_pair = mat.numpy()[a1, p]  # Convert to NumPy for indexing
        neg_pair = mat.numpy()[a2, n]  # Convert to NumPy for indexing
        self.set_stats(pos_pair, neg_pair)
        
        if self.distance.is_inverted:
            pos_mask = pos_pair < self.pos_margin
            neg_mask = neg_pair > self.neg_margin
        else:
            pos_mask = pos_pair > self.pos_margin
            neg_mask = neg_pair < self.neg_margin

        return paddle.to_tensor(a1[pos_mask]), paddle.to_tensor(p[pos_mask]), paddle.to_tensor(a2[neg_mask]), paddle.to_tensor(n[neg_mask])

    def set_stats(self, pos_pair, neg_pair):
        if self.collect_stats:
            self.pos_pair_dist = float(pos_pair.mean()) if len(pos_pair) > 0 else 0
            self.neg_pair_dist = float(neg_pair.mean()) if len(neg_pair) > 0 else 0
