"""Fit the v1 happiness model on GSS, build the poststrat seed, and estimate per-tract happiness.

v1 model = M4 (circumstantial + general health), binned (binning.py). Two simple, robust fits on the
GSS cells: a binary logit for `pct_very_happy` and an OLS on the 0/50/100 score for `happiness_index`.
Both predicted per seed cell, then poststratified (raked) onto each tract's ACS+PLACES margins.

The PLACES module-pooling problem (M5's mental/smoking live in disjoint GSS years) does NOT bite here:
health is asked in most years, so M4 fits on a large complete-case sample. M5 + uncertainty are v2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microhappiness.binning import PREDICTORS
from microhappiness.poststratify import rake

_FORMULA_RHS = "married + C(employment) + home_owner + lives_alone + income4 + I(income4**2) + health"


def fit_models(gss_binned):
    """Fit the binary very-happy logit + the 0-100 index OLS on complete GSS cells. Returns (logit, ols)."""
    import statsmodels.formula.api as smf

    d = gss_binned.dropna(subset=["happy", *PREDICTORS]).copy()
    d["very_happy"] = (d["happy"] == 3).astype(int)
    d["score"] = d["happy"].map({3: 100.0, 2: 50.0, 1: 0.0})
    logit = smf.logit(f"very_happy ~ {_FORMULA_RHS}", data=d).fit(disp=0)
    ols = smf.ols(f"score ~ {_FORMULA_RHS}", data=d).fit()
    return logit, ols, len(d)


def build_seed(gss_binned):
    """GSS joint over the binned predictors -> (cells DataFrame, weight vector, per-cell predictions src).

    Returns a DataFrame with one row per observed cell (the predictor columns) and a normalised weight.
    """
    d = gss_binned.dropna(subset=PREDICTORS).copy()
    wcol = "wtssps" if "wtssps" in d and d["wtssps"].notna().any() else None
    d["_w"] = d[wcol].fillna(1.0) if wcol else 1.0
    seed = d.groupby(list(PREDICTORS), as_index=False)["_w"].sum()
    seed["_w"] /= seed["_w"].sum()
    return seed


def predict_cells(seed, logit, ols):
    """Attach pct_very_happy + happiness_index predictions to each seed cell."""
    seed = seed.copy()
    seed["pct_very"] = logit.predict(seed) * 100.0
    seed["index"] = ols.predict(seed)
    return seed


def estimate_tract(seed_cols, w0, preds, acs_margin, health_fraction):
    """Rake the seed to one tract's margins (+ health) and return its happiness estimates."""
    margins = dict(acs_margin)
    margins["health"] = {1.0: health_fraction, 0.0: 1.0 - health_fraction}
    w = rake(seed_cols, w0, margins)
    return {
        "happiness_index": float(np.dot(w, preds["index"].to_numpy())),
        "pct_very_happy": float(np.dot(w, preds["pct_very"].to_numpy())),
    }


def estimate_state(gss_binned, acs_margins, places_health):
    """Estimate every tract that has both ACS margins and a PLACES health value. -> DataFrame."""
    logit, ols, n_fit = fit_models(gss_binned)
    seed = predict_cells(build_seed(gss_binned), logit, ols)
    seed_cols = {p: seed[p].to_numpy() for p in PREDICTORS}
    w0 = seed["_w"].to_numpy()
    rows = []
    for geoid, margin in acs_margins.items():
        h = places_health.get(geoid)
        if not h:
            continue
        est = estimate_tract(seed_cols, w0, seed, margin, h["fraction"])
        rows.append({"geoid": geoid, **est, "adult_pop": h["adult_pop"]})
    return pd.DataFrame(rows), {"n_fit": n_fit, "n_cells": len(seed)}
