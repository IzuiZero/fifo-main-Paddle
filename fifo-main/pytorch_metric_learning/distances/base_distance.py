import paddle
import paddle.nn as nn

from ..utils.module_with_records import ModuleWithRecords  # Assuming ModuleWithRecords is correctly imported

class BaseDistance(ModuleWithRecords):
    def __init__(self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(list_of_names=["p", "power"], is_stat=False)

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(query_emb, ref_emb, query_emb_normalized, ref_emb_normalized)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        assert mat.shape == paddle.to_tensor(query_emb.shape[0], ref_emb.shape[0]).shape
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return paddle.max(*args, **kwargs)
        return paddle.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return paddle.min(*args, **kwargs)
        return paddle.max(*args, **kwargs)

    # This measures the margin between x and y
    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, axis=1, **kwargs):
        return nn.functional.normalize(embeddings, p=self.p, axis=axis, **kwargs)

    def maybe_normalize(self, embeddings, axis=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, axis=axis, **kwargs)
        return embeddings

    def get_norm(self, embeddings, axis=1, **kwargs):
        return paddle.norm(embeddings, p=self.p, axis=axis, **kwargs)

    def set_default_stats(self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized):
        if self.collect_stats:
            with paddle.no_grad():
                stats_dict = {
                    "initial_avg_query_norm": paddle.mean(self.get_norm(query_emb)).item(),
                    "initial_avg_ref_norm": paddle.mean(self.get_norm(ref_emb)).item(),
                    "final_avg_query_norm": paddle.mean(self.get_norm(query_emb_normalized)).item(),
                    "final_avg_ref_norm": paddle.mean(self.get_norm(ref_emb_normalized)).item(),
                }
                self.set_stats(stats_dict)

    def set_stats(self, stats_dict):
        for k, v in stats_dict.items():
            self.add_to_recordable_attributes(name=k, is_stat=True)
            setattr(self, k, v)
