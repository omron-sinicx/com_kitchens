import scipy.stats
import torch


def rank_tensor(t: torch.Tensor, dim=-1, method: str = None) -> torch.Tensor:
    assert method in (None, "average", "min", "max", "dense", "ordinal")

    if method is None:
        return t.argsort(dim=dim).argsort(dim=dim)
    else:
        # ranks by scipy.stats.rankdata starts from 1 (0 for torch.argsort)
        ranks = scipy.stats.rankdata(t.cpu(), axis=dim, method=method) - 1
        return torch.Tensor(ranks).to(dtype=torch.int64).to(t.device)


def masked_mean(t: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    # cannot fill LongTensor with nan
    t = t.float()
    return t.masked_fill(~mask, torch.nan).nanmean(dim=dim, keepdim=keepdim)


def masked_median(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # cannot fill LongTensor with nan
    t = t.float()
    return t.masked_fill(~mask, torch.nan).nanmedian()


def scatter_min_2d_label(samples: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    """Select mean(samples), count() from samples group by labels order by labels asc."""

    assert samples.size(0) == labels.size(0)

    unique_labels, inverse_labels = labels.unique(dim=0, return_inverse=True)

    res_shape = unique_labels.shape[:1] + samples.shape[1:]
    res = torch.full(
        res_shape, samples.max(), dtype=samples.dtype, device=samples.device
    ).scatter_reduce(
        dim=0,
        index=inverse_labels.view(-1, 1).repeat(1, samples.size(-1)),
        src=samples,
        reduce="amin",
    )

    return res, unique_labels
