import paddle
from .base_miner import BaseTupleMiner


class EmbeddingsAlreadyPackagedAsTriplets(BaseTupleMiner):
    # If the embeddings are grouped by triplet,
    # then use this miner to force the loss function to use the already-formed triplets
    def mine(self, embeddings, labels, ref_emb, ref_labels):
        batch_size = embeddings.shape[0]
        a = paddle.arange(0, batch_size, 3)
        p = paddle.arange(1, batch_size, 3)
        n = paddle.arange(2, batch_size, 3)
        return a, p, n
