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
from microhappiness.poststratify import precompute_masks, rake

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


def estimate_state(gss_binned, acs_margins, places_health, *, fitted=None):
    """Estimate every area that has both ACS margins and a PLACES health value. -> DataFrame.

    `fitted` = (seed, masks, w0, idx, pv, n_fit) to reuse the GSS fit across states (built if None).
    """
    if fitted is None:
        logit, ols, n_fit = fit_models(gss_binned)
        seed = predict_cells(build_seed(gss_binned), logit, ols)
        masks = precompute_masks({p: seed[p].to_numpy() for p in PREDICTORS})
        fitted = (seed, masks, seed["_w"].to_numpy(),
                  seed["index"].to_numpy(), seed["pct_very"].to_numpy(), n_fit)
    seed, masks, w0, idx, pv, n_fit = fitted
    rows = []
    for geoid, margin in acs_margins.items():
        h = places_health.get(geoid)
        if not h:
            continue
        m = dict(margin)
        m["health"] = {1.0: h["fraction"], 0.0: 1.0 - h["fraction"]}
        w = rake(masks, w0, m)
        rows.append({"geoid": geoid, "happiness_index": float(np.dot(w, idx)),
                     "pct_very_happy": float(np.dot(w, pv)), "adult_pop": h["adult_pop"]})
    return pd.DataFrame(rows), {"n_fit": n_fit, "n_cells": len(seed)}


# ----- M5: add the mental-health + smoking PLACES margins -----------------------------------------
# The two extra predictors live in disjoint GSS year-modules (no complete cases with the M4 set), so
# we GRAFT: estimate each one's coefficient on its OWN co-observed sample, add it to the M4 linear
# predictor, and extend the seed by conditional independence given health (P(extra | health) from the
# observed sub-sample). Honest about its limits: mental adds ~+0.013, smoking ~+0.004 (a level marker).
M5_EXTRA = ("mental_health", "smoker")


def fit_models_m5(gss_binned):
    """Return the M4 fits + grafted {extra: (logit_coef, ols_coef)} + P(extra=1 | health)."""
    import statsmodels.formula.api as smf

    d = gss_binned.copy()
    d["very_happy"] = (d["happy"] == 3).astype(int)
    d["score"] = d["happy"].map({3: 100.0, 2: 50.0, 1: 0.0})
    logit, ols, n_fit = fit_models(gss_binned)
    coef, cond = {}, {}
    for extra in M5_EXTRA:
        sub = d.dropna(subset=["very_happy", *PREDICTORS, extra])
        lg = smf.logit(f"very_happy ~ {_FORMULA_RHS} + {extra}", data=sub).fit(disp=0)
        ol = smf.ols(f"score ~ {_FORMULA_RHS} + {extra}", data=sub).fit()
        coef[extra] = (float(lg.params[extra]), float(ol.params[extra]))
        cobs = d.dropna(subset=["health", extra])
        cond[extra] = {h: float(cobs.loc[cobs["health"] == h, extra].mean()) for h in (0.0, 1.0)}
    return logit, ols, coef, cond, n_fit


def build_seed_m5(gss_binned, logit, ols, coef, cond):
    """Extend the M4 seed with mental/smoking dimensions (cond. independent given health) + predictions."""
    base = build_seed(gss_binned)
    base["_lin"] = logit.predict(base, which="linear")   # M4 logit linear predictor
    base["_idx"] = ols.predict(base)                  # M4 index
    out = []
    for _, c in base.iterrows():
        h = c["health"]
        for mh in (0.0, 1.0):
            for sm in (0.0, 1.0):
                pm = cond["mental_health"][h] if mh == 1 else 1 - cond["mental_health"][h]
                ps = cond["smoker"][h] if sm == 1 else 1 - cond["smoker"][h]
                lin = c["_lin"] + coef["mental_health"][0] * mh + coef["smoker"][0] * sm
                idx = c["_idx"] + coef["mental_health"][1] * mh + coef["smoker"][1] * sm
                row = {p: c[p] for p in PREDICTORS}
                row.update(mental_health=mh, smoker=sm, _w=c["_w"] * pm * ps,
                           pct_very=100.0 / (1.0 + np.exp(-lin)), index=idx)
                out.append(row)
    return pd.DataFrame(out)


def fit_m5(gss_binned):
    """Build the reusable M5 fit (seed masks + per-cell predictions) once, to share across states."""
    logit, ols, coef, cond, n_fit = fit_models_m5(gss_binned)
    seed = build_seed_m5(gss_binned, logit, ols, coef, cond)
    masks = precompute_masks({p: seed[p].to_numpy() for p in (*PREDICTORS, *M5_EXTRA)})
    return (masks, seed["_w"].to_numpy(), seed["index"].to_numpy(), seed["pct_very"].to_numpy(),
            n_fit, len(seed), coef)


def estimate_state_m5(gss_binned, acs_margins, places, *, fitted=None):
    """M5 estimates. `places` = {'GHLTH':..., 'MHLTH':..., 'CSMOKING':...} of {geoid:{fraction,adult_pop}}."""
    masks, w0, ix, pv, n_fit, n_cells, coef = fitted or fit_m5(gss_binned)
    rows = []
    for geoid, margin in acs_margins.items():
        gh, mh, sm = (places["GHLTH"].get(geoid), places["MHLTH"].get(geoid), places["CSMOKING"].get(geoid))
        if not (gh and mh and sm):
            continue
        m = dict(margin)
        m["health"] = {1.0: gh["fraction"], 0.0: 1 - gh["fraction"]}
        m["mental_health"] = {1.0: mh["fraction"], 0.0: 1 - mh["fraction"]}
        m["smoker"] = {1.0: sm["fraction"], 0.0: 1 - sm["fraction"]}
        w = rake(masks, w0, m)
        rows.append({"geoid": geoid, "happiness_index": float(np.dot(w, ix)),
                     "pct_very_happy": float(np.dot(w, pv)), "adult_pop": gh["adult_pop"]})
    return pd.DataFrame(rows), {"n_fit": n_fit, "n_cells": n_cells, "graft": coef}
