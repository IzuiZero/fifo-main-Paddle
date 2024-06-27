import copy
import numpy as np
import paddle

from ..distances import CosineSimilarity
from . import common_functions as c_f


class MatchFinder:
    def __init__(self, distance, threshold=None):
        self.distance = distance
        self.threshold = threshold

    def operate_on_emb(self, input_func, query_emb, ref_emb=None, *args, **kwargs):
        if ref_emb is None:
            ref_emb = query_emb
        return input_func(query_emb, ref_emb, *args, **kwargs)

    # for a batch of queries
    def get_matching_pairs(
        self, query_emb, ref_emb=None, threshold=None, return_tuples=False
    ):
        with paddle.no_grad():
            threshold = threshold if threshold is not None else self.threshold
            return self.operate_on_emb(
                self._get_matching_pairs, query_emb, ref_emb, threshold, return_tuples
            )

    def _get_matching_pairs(self, query_emb, ref_emb, threshold, return_tuples):
        mat = self.distance(query_emb, ref_emb)
        matches = mat >= threshold if self.distance.is_inverted else mat <= threshold
        matches = matches.cpu().numpy()
        if return_tuples:
            return list(zip(*np.where(matches)))
        return matches

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        with paddle.no_grad():
            dist = self.distance.pairwise_distance(x, y)
            output = (
                dist >= threshold if self.distance.is_inverted else dist <= threshold
            )
            if output.nelement() == 1:
                return output.detach().item()
            return output.cpu().numpy()


class FaissIndexer:
    def __init__(self, index=None):
        import faiss as faiss_module

        self.faiss_module = faiss_module
        self.index = index

    def train_index(self, embeddings):
        self.index = self.faiss_module.IndexFlatL2(embeddings.shape[1])
        self.add_to_index(embeddings)

    def add_to_index(self, embeddings):
        self.index.add(embeddings)

    def search_nn(self, query_batch, k):
        D, I = self.index.search(query_batch, k)
        return I, D

    def save(self, filename):
        self.faiss_module.write_index(self.index, filename)

    def load(self, filename):
        self.index = self.faiss_module.read_index(filename)


class InferenceModel:
    def __init__(
        self,
        trunk,
        embedder=None,
        match_finder=None,
        normalize_embeddings=True,
        indexer=None,
    ):
        self.trunk = trunk
        self.embedder = c_f.Identity() if embedder is None else embedder
        self.match_finder = (
            MatchFinder(distance=CosineSimilarity(), threshold=0.9)
            if match_finder is None
            else match_finder
        )
        self.indexer = FaissIndexer() if indexer is None else indexer
        self.normalize_embeddings = normalize_embeddings

    def get_embeddings_from_tensor_or_dataset(self, inputs, batch_size):
        if isinstance(inputs, list):
            inputs = paddle.stack(inputs)

        embeddings = []
        if paddle.is_tensor(inputs):
            for i in range(0, len(inputs), batch_size):
                embeddings.append(self.get_embeddings(inputs[i : i + batch_size]))
        elif isinstance(inputs, paddle.io.Dataset):
            dataloader = paddle.io.DataLoader(inputs, batch_size=batch_size)
            for inp, _ in dataloader:
                embeddings.append(self.get_embeddings(inp))
        else:
            raise TypeError(f"Indexing {type(inputs)} is not supported.")
        return paddle.concat(embeddings)

    def train_indexer(self, inputs, batch_size=64):
        embeddings = self.get_embeddings_from_tensor_or_dataset(inputs, batch_size)
        self.indexer.train_index(embeddings.cpu().numpy())

    def add_to_indexer(self, inputs, batch_size=64):
        embeddings = self.get_embeddings_from_tensor_or_dataset(inputs, batch_size)
        self.indexer.add_to_index(embeddings.cpu().numpy())

    def get_nearest_neighbors(self, query, k):
        if not self.indexer.index or not self.indexer.index.is_trained:
            raise RuntimeError("Index must be trained by running `train_indexer`")

        query_emb = self.get_embeddings(query)

        indices, distances = self.indexer.search_nn(query_emb.cpu().numpy(), k)
        return indices, distances

    def get_embeddings(self, x):
        if isinstance(x, list):
            x = paddle.stack(x)

        self.trunk.eval()
        self.embedder.eval()
        with paddle.no_grad():
            x_emb = self.embedder(self.trunk(x))
        if self.normalize_embeddings:
            x_emb = paddle.nn.functional.normalize(x_emb, p=2, axis=1)
        return x_emb

    # for a batch of queries
    def get_matches(self, query, ref=None, threshold=None, return_tuples=False):
        query_emb = self.get_embeddings(query)
        ref_emb = query_emb
        if ref is not None:
            ref_emb = self.get_embeddings(ref)
        return self.match_finder.get_matching_pairs(
            query_emb, ref_emb, threshold, return_tuples
        )

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        x = self.get_embeddings(x)
        y = self.get_embeddings(y)
        return self.match_finder.is_match(x, y, threshold)

    def save_index(self, filename):
        self.indexer.save(filename)

    def load_index(self, filename):
        self.indexer.load(filename)


class LogitGetter(paddle.nn.Layer):
    possible_layer_names = ["fc", "proxies", "W"]

    def __init__(
        self,
        classifier,
        layer_name=None,
        transpose=None,
        distance=None,
        copy_weights=True,
    ):
        super().__init__()
        self.copy_weights = copy_weights
        ### set layer weights ###
        if layer_name is not None:
            self.set_weights(getattr(classifier, layer_name))
        else:
            for x in self.possible_layer_names:
                layer = getattr(classifier, x, None)
                if layer is not None:
                    self.set_weights(layer)
                    break

        ### set distance measure ###
        self.distance = classifier.distance if distance is None else distance
        self.transpose = transpose

    def forward(self, embeddings):
        w = self.weights
        if self.transpose is True:
            w = w.t()
        elif self.transpose is None:
            if w.shape[0] == embeddings.shape[1]:
                w = w.t()
        return self.distance(embeddings, w)

    def set_weights(self, layer):
        self.weights = copy.deepcopy(layer) if self.copy_weights else layer
