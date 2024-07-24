from typing import List

import numpy as np
import torch
from torch import Tensor

from com_kitchens.utils.pytorch import (
    masked_mean,
    masked_median,
    rank_tensor,
    scatter_min_2d_label,
)


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics["R1"] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics["R5"] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics["R10"] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics["MR"] = np.median(ind) + 1
    metrics["MedianR"] = metrics["MR"]
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics["R1"]
    r5 = metrics["R5"]
    r10 = metrics["R10"]
    mr = metrics["MR"]
    print(f"R@1: {r1:.4f} - R@5: {r5:.4f} - R@10: {r10:.4f} - Median R: {mr}")


# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k=[1, 5, 10]):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
    second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=2))
    mask = ~torch.logical_or(
        torch.isinf(permuted_original_data), torch.isnan(permuted_original_data)
    )
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    # assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
        valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results["MR"] = results["MedianR"]
    return results


def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float("-inf")
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T


# retrieval metrics
def v2t_score(
    sim_logits: Tensor,
    matched_mask: Tensor,
    pool_mask: Tensor = None,
    at=1,
):
    if pool_mask is not None:
        sim_logits = sim_logits.masked_fill(~pool_mask, torch.inf)

    # ranks for each pair (same rank for equivalents)
    ranks = rank_tensor(sim_logits, method="min")

    # score
    if at == "mean":
        m_score = masked_mean(ranks, matched_mask)
    elif at == "median":
        # m_score = masked_median(ranks, matched_mask)
        min_ranks = ranks.masked_fill(~matched_mask, ranks.max() + 1).min(dim=-1)[0]
        m_score = float(torch.median(min_ranks))
    else:
        # minimum rank for each video
        min_ranks = ranks.masked_fill(~matched_mask, ranks.max() + 1).min(dim=-1)[0]
        m_score = float(sum(min_ranks < at)) * 100 / len(min_ranks)

    return m_score


def get_matched_matrix(
    seq_ids: Tensor,
    vid_ids: Tensor,
):
    n_vid = vid_ids.size(0)
    n_seq = seq_ids.size(0)

    vid_ids = vid_ids.unsqueeze(1).repeat(1, n_seq, 1)
    seq_ids = seq_ids.unsqueeze(0).repeat(n_vid, 1, 1)

    match_matrix = (vid_ids == seq_ids).all(dim=-1)

    return match_matrix


def get_matched_matrix_from_json(
    seq_ids: Tensor,
    vid_ids: Tensor,
    stage: str,
):
    n_vid = vid_ids.size(0)
    n_seq = seq_ids.size(0)

    import json

    json_path = f"/workspace/com_kitchens/data/main/feasible_{stage}.json"
    with open(json_path) as f:
        feasible_recipe = json.load(f)

    match_matrix = torch.zeros(n_vid, n_seq, dtype=torch.bool)
    for i in range(n_vid):
        for j in range(n_seq):
            vid_id_str = str(vid_ids[i][0].item()) + "_" + str(vid_ids[i][1].item())
            seq_id_str = str(seq_ids[j][0].item()) + "_" + str(seq_ids[j][1].item())
            if vid_id_str == seq_id_str:
                match_matrix[i][j] = True
            elif vid_id_str in feasible_recipe and seq_id_str in feasible_recipe[vid_id_str]:
                match_matrix[i][j] = True

    match_matrix = match_matrix.to(seq_ids.device)
    return match_matrix


def v2t_m1_score(
    sim_logits: Tensor,
    seq_ids: Tensor,
    vid_ids: Tensor,
    at=1,
):
    # reduce sim_logits to recipe/kitchen-granularity
    agg_sim_logits, agg_seq_ids = scatter_min_2d_label(sim_logits.T, seq_ids[:, :2])
    agg_sim_logits = agg_sim_logits.T

    vid_ids = vid_ids[:, :2]
    matched_matrix = get_matched_matrix(agg_seq_ids, vid_ids)

    return v2t_score(
        sim_logits=agg_sim_logits,
        matched_mask=matched_matrix,
        at=at,
    )


def v2t_FEASIBLE_score(
    sim_logits: Tensor,
    seq_ids: Tensor,
    vid_ids: Tensor,
    at=1,
    stage="early",
):
    # reduce sim_logits to recipe/kitchen-granularity
    agg_sim_logits, agg_seq_ids = scatter_min_2d_label(sim_logits.T, seq_ids[:, :2])
    agg_sim_logits = agg_sim_logits.T

    vid_ids = vid_ids[:, :2]
    matched_matrix = get_matched_matrix_from_json(agg_seq_ids, vid_ids, stage=stage)

    return v2t_score(
        sim_logits=agg_sim_logits,
        matched_mask=matched_matrix,
        at=at,
    )


def v2t_m2_score(
    sim_logits: Tensor,
    seq_ids: Tensor,
    vid_ids: Tensor,
    at=1,
):
    pool_mask = get_matched_matrix(seq_ids[:, :2], vid_ids[:, :2])
    matched_mask = get_matched_matrix(seq_ids, vid_ids)

    return v2t_score(
        sim_logits=sim_logits,
        pool_mask=pool_mask,
        matched_mask=matched_mask,
        at=at,
    )


def v2t_m3_score(
    sim_logits: Tensor,
    seq_ids: Tensor,
    vid_ids: Tensor,
    at=1,
):
    matched_mask = get_matched_matrix(seq_ids, vid_ids)

    return v2t_score(
        sim_logits=sim_logits,
        matched_mask=matched_mask,
        at=at,
    )


def compute_scores(
    sim_matrix: Tensor,
    sequence_ids: Tensor,
    video_ids: Tensor,
    stage: str = "early",
):
    metrics = {}

    metrics["M1-R1"] = v2t_m1_score(sim_matrix, sequence_ids, video_ids, at=1)
    metrics["M1-R5"] = v2t_m1_score(sim_matrix, sequence_ids, video_ids, at=5)
    metrics["M1-R10"] = v2t_m1_score(sim_matrix, sequence_ids, video_ids, at=10)
    # metrics["M1-mean"] = v2t_m1_score(sim_matrix, sequence_ids, video_ids, at="mean")
    metrics["M1-median"] = v2t_m1_score(sim_matrix, sequence_ids, video_ids, at="median")

    # As feasible recipe retrieval requires a private dataset CRD, we just ignore the metrics.
    # metrics["FEAS-R1"] = v2t_FEASIBLE_score(sim_matrix, sequence_ids, video_ids, at=1, stage=stage)
    # metrics["FEAS-R5"] = v2t_FEASIBLE_score(sim_matrix, sequence_ids, video_ids, at=5, stage=stage)
    # metrics["FEAS-R10"] = v2t_FEASIBLE_score(
    #     sim_matrix, sequence_ids, video_ids, at=10, stage=stage
    # )
    # # metrics["FEAS-R50"] = v2t_FEASIBLE_score(sim_matrix, sequence_ids, video_ids, at=50)
    # # metrics["FEAS-R100"] = v2t_FEASIBLE_score(sim_matrix, sequence_ids, video_ids, at=100)
    # metrics["FEAS-median"] = v2t_FEASIBLE_score(
    #     sim_matrix, sequence_ids, video_ids, at="median", stage=stage
    # )

    metrics["M2-R1"] = v2t_m2_score(sim_matrix, sequence_ids, video_ids, at=1)
    metrics["M2-R5"] = v2t_m2_score(sim_matrix, sequence_ids, video_ids, at=5)
    metrics["M2-R10"] = v2t_m2_score(sim_matrix, sequence_ids, video_ids, at=10)
    # metrics["M2-mean"] = v2t_m2_score(sim_matrix, sequence_ids, video_ids, at="mean")
    metrics["M2-median"] = v2t_m2_score(sim_matrix, sequence_ids, video_ids, at="median")

    # metrics["M3-R1"] = v2t_m3_score(sim_matrix, sequence_ids, video_ids, at=1)
    # metrics["M3-R5"] = v2t_m3_score(sim_matrix, sequence_ids, video_ids, at=5)
    # metrics["M3-R10"] = v2t_m3_score(sim_matrix, sequence_ids, video_ids, at=10)
    # metrics["M3-mean"] = v2t_m3_score(sim_matrix, sequence_ids, video_ids, at="mean")
    # metrics["M3-median"] = v2t_m3_score(sim_matrix, sequence_ids, video_ids, at="median")

    return metrics
