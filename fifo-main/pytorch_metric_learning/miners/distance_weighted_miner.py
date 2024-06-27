import paddle
from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseTupleMiner

class DistanceWeightedMiner(BaseTupleMiner):
    def __init__(self, cutoff=0.5, nonzero_loss_cutoff=1.4, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.cutoff = float(cutoff)
        self.nonzero_loss_cutoff = float(nonzero_loss_cutoff)
        self.add_to_recordable_attributes(
            list_of_names=["cutoff", "nonzero_loss_cutoff"], is_stat=False
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        dtype = embeddings.dtype
        d = float(embeddings.shape[1])
        mat = self.distance(embeddings, ref_emb)

        # Cut off to avoid high variance.
        mat = paddle.clip(mat, min=self.cutoff)

        # See the first equation from Section 4 of the paper
        log_weights = (2.0 - d) * paddle.log(mat) - ((d - 3) / 2) * paddle.log(
            1.0 - 0.25 * (mat ** 2.0)
        )

        inf_or_nan = paddle.isinf(log_weights) | paddle.isnan(log_weights)

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = paddle.ones_like(log_weights)
        same_class = labels.unsqueeze(1) == ref_labels.unsqueeze(0)
        mask[same_class] = 0
        log_weights = log_weights * mask
        # Subtract max(log(distance)) for stability.
        weights = paddle.exp(log_weights - paddle.max(log_weights[~inf_or_nan]))

        weights = (
            weights * mask * (c_f.to_dtype(mat < self.nonzero_loss_cutoff, dtype=dtype))
        )
        weights[inf_or_nan] = 0

        weights = weights / paddle.sum(weights, axis=1, keepdim=True)

        return lmu.get_random_triplet_indices(
            labels.numpy(), ref_labels=ref_labels.numpy(), weights=weights.numpy()
        )
