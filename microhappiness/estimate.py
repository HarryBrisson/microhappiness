"""Fit a candidate model on GSS, predict each poststrat cell, aggregate to area estimates + CIs.

The MRP core: P(happy | cell) from the survey model, weighted by the cell's ACS population, summed
per area. Uncertainty by Monte-Carlo over the fitted model (PLACES draws ~1,000 datasets); GSS's
smaller sample yields wider intervals — let those set the minimum credible reporting geography.
"""

from __future__ import annotations

from microhappiness.models import INDEX_CODING, ModelSpec


def fit(spec: ModelSpec, gss_df, *, temporal: bool = False):
    """Fit `spec` on recoded GSS microdata (ordinal/proportional-odds + binary very-happy).

    With temporal=True, add year random effects + marital/income x era interactions and fit the
    pooled 1972-present series so coefficients can drift by era (TEMPORAL_OVERLAY).
    """
    raise NotImplementedError("fit ordinal + binary logit with GSS weights; optional era interactions")


def predict_cells(model, poststrat_table, area_covariates=None):
    """Predicted P(very happy) and E[happiness_index] per poststrat cell per area."""
    raise NotImplementedError("apply fitted model to each cell; join area covariates for M2")


def aggregate_to_areas(cell_predictions, cell_populations):
    """Population-weighted sum of cell predictions -> per-area estimate.

    Returns rows: {geoid, happiness_index, pct_very_happy, adult_pop}. index = E[score] on
    INDEX_CODING (0/50/100), i.e. a 0-100 happiness index.
    """
    raise NotImplementedError("sum_c P(c)*pop(c) / sum_c pop(c)")


def monte_carlo_ci(model, poststrat_table, cell_populations, *, draws: int = 1000):
    """Per-area standard errors / CIs by simulating from the fitted model's coefficient covariance."""
    raise NotImplementedError("draw coefficients, re-aggregate, take per-area percentiles")
