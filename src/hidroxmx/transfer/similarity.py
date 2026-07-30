"""Donor-similarity scoring (Path A mechanism family).

Given a **target** station and a pool of **donor** stations, return a
similarity score in ``[0, 1]`` for every donor, plus a normalised
weight vector that a training loop can use to bias the pooled loss
toward the most similar donors.

The similarity is computed on a *scaled* signature space so that
low-magnitude signatures (e.g. ``high_flow_freq`` in ``[0, 1]``) do
not get overwhelmed by high-magnitude ones (e.g. ``mean_flow`` in
``[0, 500]``). We standardise on the pool statistics (target + donors
together) so the target is always at the same relative position.

Two selection modes are exposed:

- ``top_k`` — keep the K most similar donors, set the rest to zero.
- ``soft`` — assign every donor a weight proportional to a
  temperature-scaled softmax of the similarity, so all donors
  contribute but the most similar dominate.

The default mode is ``soft`` because it preserves the full data volume
that lumped multi-station training (F0-PUB) already validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class SimilarityResult:
    """Per-donor similarity and normalised weight."""
    donor_claves: tuple[str, ...]
    raw_scores: np.ndarray          # cosine or 1/(1+distance) in [0, 1]
    weights: np.ndarray             # sums to len(donor_claves) so mean_i w_i == 1
    method: str                     # 'soft' or 'top_k'
    metric: str                     # 'euclidean' or 'cosine'


def _standardise_pool(target_vec: np.ndarray,
                      donor_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both target and donor vectors against pool stats.

    NaN entries in a signature column are treated as zero AFTER scaling
    so the coordinate simply drops out of the distance for that donor;
    the alternative (imputing per-column mean) would silently pull
    NaN-heavy donors toward the pool centre.
    """
    all_vecs = np.vstack([target_vec[None, :], donor_vecs])
    mu = np.nanmean(all_vecs, axis=0)
    sigma = np.nanstd(all_vecs, axis=0, ddof=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    scaled = (all_vecs - mu) / sigma
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return scaled[0], scaled[1:]


def _euclidean_similarity(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Map Euclidean distance to a similarity in ``(0, 1]``.

    ``sim = 1 / (1 + distance)``. Not linear but monotonic and stable
    when many donors sit at similar distances.
    """
    diff = d - t[None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return 1.0 / (1.0 + dist)


def _cosine_similarity(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Cosine similarity in ``[−1, 1]`` shifted to ``[0, 1]``."""
    tn = np.linalg.norm(t) + 1e-12
    dn = np.linalg.norm(d, axis=1) + 1e-12
    cos = (d @ t) / (dn * tn)
    return 0.5 * (cos + 1.0)


def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax that yields weights summing to len(scores).

    The lumped-baseline case (all donors weighted equally) corresponds
    to ``temperature → ∞`` where softmax degenerates to uniform. Lower
    temperatures concentrate mass on the most similar donors.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = scores / temperature
    z = z - z.max()  # numerical stability
    exp = np.exp(z)
    p = exp / exp.sum()
    return p * len(scores)  # rescale so mean weight = 1


def _top_k_weights(scores: np.ndarray, k: int) -> np.ndarray:
    """Mask out all but the top-k donors, weight them uniformly at 1."""
    n = len(scores)
    k = max(1, min(int(k), n))
    idx = np.argsort(scores)[::-1][:k]
    w = np.zeros(n, dtype=float)
    w[idx] = n / k  # so mean weight = 1
    return w


def score_donors(target_vec: np.ndarray,
                 donor_vecs: np.ndarray,
                 donor_claves: Sequence[str],
                 *,
                 method: str = "soft",
                 metric: str = "euclidean",
                 temperature: float = 1.0,
                 top_k: int = 5) -> SimilarityResult:
    """Score every donor against the target.

    Parameters
    ----------
    target_vec, donor_vecs : signature vectors already assembled with
        :func:`hidroxmx.transfer.signatures.signature_vector`.
    method : ``'soft'`` uses temperature-scaled softmax weights;
        ``'top_k'`` keeps only the ``top_k`` most similar donors.
    metric : ``'euclidean'`` (default, physical-scale sensitive after
        z-scoring) or ``'cosine'`` (shape-only, scale-invariant).
    temperature : softmax temperature — higher = closer to uniform.
    top_k : number of donors to keep when ``method='top_k'``.
    """
    t_std, d_std = _standardise_pool(target_vec, donor_vecs)
    if metric == "euclidean":
        raw = _euclidean_similarity(t_std, d_std)
    elif metric == "cosine":
        raw = _cosine_similarity(t_std, d_std)
    else:
        raise ValueError(f"unknown metric {metric!r}")

    if method == "soft":
        w = _softmax_weights(raw, temperature)
    elif method == "top_k":
        w = _top_k_weights(raw, top_k)
    else:
        raise ValueError(f"unknown method {method!r}")

    return SimilarityResult(
        donor_claves=tuple(donor_claves),
        raw_scores=raw,
        weights=w,
        method=method,
        metric=metric,
    )
