"""Per-area joint poststratification tables via iterative proportional fitting (raking).

PUMS microdata is only identified to PUMA, so we can't read a tract's joint demographic distribution
directly. IPF/raking synthesizes it: take a disaggregate seed (PUMS, or a national joint) and adjust
it to match each area's published ACS marginals (gss-aligned categories). PopulationSim / RTI
SynthPop are production prior art for exactly this.

M2 (PLACES-analog) skips this — it uses census published joint cells + area covariates instead.
"""

from __future__ import annotations


def rake(seed, marginals, *, max_iter: int = 50, tol: float = 1e-6):
    """Fit a joint cell table to a set of marginal controls by IPF.

    seed: array of joint cell weights (the national/PUMS prior shape).
    marginals: list of (axis, target_vector) controls from ACS for one area.
    Returns the raked joint table whose every margin matches its ACS control (within tol).
    Pitfalls to handle: zero-marginal cells, non-convergence, and small-area instability — log and
    fall back to the area's parent (county/PUMA) joint when a tract's marginals are too sparse.
    """
    raise NotImplementedError("IPF: scale seed to each marginal in turn until margins converge")


def build_poststrat_table(area_marginals, seed):
    """{GEOID -> joint cell table} for every area, raking the shared seed to each area's marginals."""
    raise NotImplementedError("loop areas -> rake(seed, area_marginals[geoid])")
