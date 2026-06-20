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
        prev = w
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
