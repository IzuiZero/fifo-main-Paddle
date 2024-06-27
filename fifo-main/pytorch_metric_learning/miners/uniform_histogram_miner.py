import paddle

from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseTupleMiner


class UniformHistogramMiner(BaseTupleMiner):
    def __init__(self, num_bins=100, pos_per_bin=10, neg_per_bin=10, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.pos_per_bin = pos_per_bin
        self.neg_per_bin = neg_per_bin
        self.add_to_recordable_attributes(
            list_of_names=["pos_per_bin", "neg_per_bin"], is_stat=False
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels.numpy(), ref_labels.numpy())
        pos_pairs = mat.numpy()[a1, p]
        neg_pairs = mat.numpy()[a2, n]

        if len(pos_pairs) > 0:
            a1, p = self.get_uniformly_distributed_pairs(
                paddle.to_tensor(pos_pairs), a1, p, self.pos_per_bin
            )

        if len(neg_pairs) > 0:
            a2, n = self.get_uniformly_distributed_pairs(
                paddle.to_tensor(neg_pairs), a2, n, self.neg_per_bin
            )

        return a1, p, a2, n

    def get_bins(self, pairs):
        device, dtype = pairs.device, pairs.dtype
        return paddle.linspace(
            paddle.min(pairs),
            paddle.max(pairs),
            num=self.num_bins + 1,
            dtype=dtype,
        )

    def filter_by_bin(self, distances, bins, num_pairs):
        range_max = len(bins) - 1
        all_idx = []
        for i in range(range_max):
            s, e = bins[i], bins[i + 1]
            low_condition = s <= distances
            high_condition = distances < e if i != range_max - 1 else distances <= e
            condition = paddle.nonzero(low_condition & high_condition, as_tuple=False)[:, 0]
            if len(condition) == 0:
                continue
            idx = paddle.multinomial(
                paddle.ones_like(condition, dtype='float32'),
                num_samples=num_pairs,
                replacement=True,
            )
            all_idx.append(condition[idx])
        return paddle.concat(all_idx, axis=0)

    def get_uniformly_distributed_pairs(self, distances, anchors, others, num_pairs):
        bins = self.get_bins(distances)
        idx = self.filter_by_bin(distances, bins, num_pairs)
        return anchors[idx], others[idx]
