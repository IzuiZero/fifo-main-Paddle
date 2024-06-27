import math
import numpy as np
import paddle

from . import common_functions as c_f


# input must be 2D
def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = paddle.masked_fill(x, ~keep_mask, c_f.neg_inf(x.dtype))
    if add_one:
        zeros = paddle.zeros((x.shape[dim - 1],), dtype=x.dtype, device=x.device).unsqueeze(dim)
        x = paddle.concat([x, zeros], axis=dim)

    output = paddle.logsumexp(x, axis=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~paddle.any(keep_mask, axis=dim, keepdim=True), 0)
    return output


def meshgrid_from_sizes(x, y, dim=0):
    a = paddle.arange(x.shape[dim], dtype=x.dtype, device=x.device)
    b = paddle.arange(y.shape[dim], dtype=y.dtype, device=y.device)
    return paddle.meshgrid(a, b)


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2)
    diffs = matches.logical_not()
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = paddle.where(matches)
    a2_idx, n_idx = paddle.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(indices_tuple, labels):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n


def convert_to_pos_pairs_with_unique_labels(indices_tuple, labels):
    a, p, _, _ = convert_to_pairs(indices_tuple, labels)
    _, unique_idx = np.unique(labels[a].cpu().numpy(), return_index=True)
    return a[unique_idx], p[unique_idx]


def pos_pairs_from_tuple(indices_tuple):
    return indices_tuple[:2]


def neg_pairs_from_tuple(indices_tuple):
    return indices_tuple[2:]


def get_all_triplets_indices(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2)
    diffs = matches.logical_not()
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return paddle.where(triplets)


# sample triplets, with a weighted distribution if weights is specified.
def get_random_triplet_indices(
    labels, ref_labels=None, t_per_anchor=None, weights=None
):
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = paddle.unique(labels)
    for label in unique_labels:
        # Get indices of positive samples for this label.
        p_inds = paddle.where(ref_labels == label)[0]
        if ref_labels is labels:
            a_inds = p_inds
        else:
            a_inds = paddle.where(labels == label)[0]
        n_inds = paddle.where(ref_labels != label)[0]
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        if ref_labels is labels:
            p_inds_ = p_inds_[~paddle.eye(n_a, dtype=paddle.bool)].view((n_a, n_a - 1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = paddle.randint(0, p_inds_.shape[1], (num_triplets,))
        # Get indices of indices of corresponding anchors.
        a_ = paddle.arange(n_a).view(-1, 1).repeat(1, k).view(num_triplets)
        p = p_inds_[a_, p_]
        a = a_inds[a_]

        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = paddle.where(paddle.sum(w, axis=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == paddle.float16:
                # special case needed due to paddle cuda bug
                w = w.astype(paddle.float32)
            n_ = paddle.multinomial(w, 1, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            n_ = paddle.randint(0, len(n_inds), (num_triplets,))
        n = n_inds[n_]
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    if len(a_idx) > 0:
        a_idx = c_f.to_device(paddle.concat(a_idx), device=labels_device, dtype=paddle.long)
        p_idx = c_f.to_device(paddle.concat(p_idx), device=labels_device, dtype=paddle.long)
        n_idx = c_f.to_device(paddle.concat(n_idx), device=labels_device, dtype=paddle.long)
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = paddle.tensor([], device=labels_device, dtype=paddle.long)
        return empty.clone(), empty.clone(), empty.clone()


def repeat_to_match_size(smaller_set, larger_size, smaller_size):
    num_repeat = math.ceil(float(larger_size) / float(smaller_size))
    return smaller_set.tile((num_repeat,))[:larger_size]


def matched_size_indices(curr_p_idx, curr_n_idx):
    num_pos_pairs = len(curr_p_idx)
    num_neg_pairs = len(curr_n_idx)
    if num_pos_pairs > num_neg_pairs:
        n_idx = repeat_to_match_size(curr_n_idx, num_pos_pairs, num_neg_pairs)
        p_idx = curr_p_idx
    else:
        p_idx = repeat_to_match_size(curr_p_idx, num_neg_pairs, num_pos_pairs)
        n_idx = curr_n_idx
    return p_idx, n_idx


def convert_to_triplets(indices_tuple, labels, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels)
        else:
            return get_random_triplet_indices(labels, t_per_anchor=t_per_anchor)
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a1, p, a2, n = indices_tuple
        p_idx, n_idx = paddle.where(a1.unsqueeze(1) == a2)
        return a1[p_idx], p[p_idx], n[n_idx]


def convert_to_weights(indices_tuple, labels, dtype):
    """
    Returns a weight for each batch element, based on
    how many times they appear in indices_tuple.
    """
    weights = paddle.zeros(labels.shape[0], device=labels.device)
    weights = c_f.to_dtype(weights, dtype=dtype)
    if (indices_tuple is None) or (all(len(x) == 0 for x in indices_tuple)):
        return weights + 1
    indices, counts = paddle.unique(paddle.concat(indices_tuple, axis=0), return_counts=True)
    counts = c_f.to_dtype(counts, dtype=dtype) / paddle.sum(counts)
    weights[indices] = counts / paddle.max(counts)
    return weights
