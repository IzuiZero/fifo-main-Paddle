import paddle
from paddle.nn.parallel import DistributedDataParallel as DDP
from ..utils import common_functions as c_f


# modified from https://github.com/allenai/allennlp
def is_distributed():
    return paddle.distributed.is_initialized()


# modified from https://github.com/JohnGiorgi/DeCLUTR
def all_gather(embeddings, labels):
    labels = c_f.to_device(labels, embeddings)
    # If we are not using distributed training, this is a no-op.
    if not is_distributed():
        return embeddings, labels
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    # Gather the embeddings on all replicas
    embeddings_list = [paddle.ones_like(embeddings) for _ in range(world_size)]
    labels_list = [paddle.ones_like(labels) for _ in range(world_size)]
    paddle.distributed.all_gather(embeddings_list, embeddings)
    paddle.distributed.all_gather(labels_list, labels)
    # The gathered copy of the current replicas embeddings have no gradients, so we overwrite
    # them with the embeddings generated on this replica, which DO have gradients.
    embeddings_list[rank] = embeddings
    labels_list[rank] = labels
    # Finally, we concatenate the embeddings
    embeddings = paddle.concat(embeddings_list)
    labels = paddle.concat(labels_list)
    return embeddings, labels


def all_gather_embeddings_labels(embeddings, labels):
    if c_f.is_list_or_tuple(embeddings):
        assert c_f.is_list_or_tuple(labels)
        all_embeddings, all_labels = [], []
        for i in range(len(embeddings)):
            E, L = all_gather(embeddings[i], labels[i])
            all_embeddings.append(E)
            all_labels.append(L)
        embeddings = paddle.concat(all_embeddings, axis=0)
        labels = paddle.concat(all_labels, axis=0)
    else:
        embeddings, labels = all_gather(embeddings, labels)

    return embeddings, labels


class DistributedLossWrapper(paddle.nn.Layer):
    def __init__(self, loss, **kwargs):
        super().__init__()
        has_parameters = len([p for p in loss.parameters()]) > 0
        self.loss = DDP(loss, **kwargs) if has_parameters else loss

    def forward(self, embeddings, labels, *args, **kwargs):
        embeddings, labels = all_gather_embeddings_labels(embeddings, labels)
        return self.loss(embeddings, labels, *args, **kwargs)


class DistributedMinerWrapper(paddle.nn.Layer):
    def __init__(self, miner):
        super().__init__()
        self.miner = miner

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        embeddings, labels = all_gather_embeddings_labels(embeddings, labels)
        if ref_emb is not None:
            ref_emb, ref_labels = all_gather_embeddings_labels(ref_emb, ref_labels)
        return self.miner(embeddings, labels, ref_emb, ref_labels)
