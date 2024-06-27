import paddle
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseTupleMiner

class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels.numpy(), ref_labels.numpy()
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat.numpy()[anchor_idx, positive_idx]
        an_dist = mat.numpy()[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        return (
            paddle.to_tensor(anchor_idx[threshold_condition]),
            paddle.to_tensor(positive_idx[threshold_condition]),
            paddle.to_tensor(negative_idx[threshold_condition]),
        )

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            self.pos_pair_dist = float(ap_dist.mean())
            self.neg_pair_dist = float(an_dist.mean())
            self.avg_triplet_margin = float(triplet_margin.mean())
