"""Per-area joint poststratification by iterative proportional fitting (raking).

We can't read a tract's joint demographic distribution directly, so we rake a national SEED (the GSS
joint over the binned predictors) to each tract's published marginals (ACS + the PLACES health margin).
The seed supplies the correlation structure (e.g. married x owner x income); the margins tilt it to the
tract. Cells absent from the seed stay at zero (we can't invent mass for combos GSS never observed).

Cell membership masks don't change across areas, so `precompute_masks` builds them once and `rake`
reuses them — the difference between minutes and hours over ~73k national tracts.
"""

from __future__ import annotations

import numpy as np


def precompute_masks(seed_cells):
    """{predictor: {level: boolean cell-membership mask}} from the seed's per-cell predictor arrays."""
    return {p: {lvl: (arr == lvl) for lvl in np.unique(arr)} for p, arr in seed_cells.items()}


def rake(masks, seed_w, margins, *, iters: int = 40, tol: float = 1e-9):
    """Rake seed weights to marginal targets using precomputed `masks`.

    masks:   from precompute_masks(seed_cells).
    seed_w:  np.array of seed proportions per cell (normalised internally).
    margins: {predictor: {level: target_proportion}} — only predictors present here are controlled.
    Returns the raked weight vector (sums to 1).
    """
    w = np.asarray(seed_w, dtype=float).copy()
    w /= w.sum()
    for _ in range(iters):
        prev = w.copy()  # a real copy — `prev = w` aliased the in-place updates, so the convergence
        # test always saw 0 and IPF silently stopped after ONE pass (fixed 2026-07)
        for pred, targets in margins.items():
            pm = masks[pred]
            for level, target in targets.items():
                mask = pm.get(level)
                if mask is None:
                    continue
                cur = w[mask].sum()
                if cur > 0 and target > 0:
                    w[mask] *= target / cur
                elif target == 0:
                    w[mask] = 0.0
            w /= w.sum()
        if np.max(np.abs(w - prev)) < tol:
            break
    return w


def rake_many(masks, seed_w, margin_list, *, iters: int = 40, tol: float = 1e-7, chunk: int = 2000):
    """Rake MANY areas at once (vectorized across areas) — the per-area python loop is the bottleneck
    once the seed grows past a few hundred cells (the identity-aware outcomes' seed is ~6k cells).

    margin_list: a list (one entry per area) of {predictor: {level: target}} dicts, all sharing the
    same predictor set and levels (missing levels are treated as target 0 for that area — consistent
    with `rake`, which zeroes cells whose target is 0).
    Returns an (n_areas x n_cells) weight matrix, rows summing to 1.
    """
    n = len(margin_list)
    c = len(seed_w)
    preds = list(margin_list[0])
    # Per predictor: level -> (n,) target vector.
    targets = {p: {lvl: np.fromiter((m[p].get(lvl, 0.0) for m in margin_list), float, n)
                   for lvl in masks[p]} for p in preds}
    out = np.empty((n, c))
    w0 = np.asarray(seed_w, dtype=float)
    w0 = w0 / w0.sum()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        w = np.tile(w0, (hi - lo, 1))
        for _ in range(iters):
            prev = w.copy()
            for p in preds:
                for lvl, mask in masks[p].items():
                    tgt = targets[p][lvl][lo:hi]
                    cur = w[:, mask].sum(axis=1)
                    # cur>0: scale to target (target 0 zeroes the cells, like `rake`); cur==0: leave.
                    factor = np.divide(tgt, cur, out=np.ones_like(cur), where=cur > 0)
                    w[:, mask] *= factor[:, None]
                w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
            if np.max(np.abs(w - prev)) < tol:
                break
        out[lo:hi] = w
    return out
