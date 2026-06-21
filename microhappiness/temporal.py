"""Temporal model: a smooth national period term + a survey-mode flag on the cross-sectional model.

The cross-sectional model tracks geography, not time (year-to-year r ~0.11). The national happiness trend
is a PERIOD effect (mood/events) the composition can't see — plus a survey-mode artifact (the 2021/2024
GSS push-to-web waves). Here period is a smooth spline over year + a web-mode dummy, so a panel can carry a
national-mood overlay on top of each area's composition.

Honest limit: we can NOT separate the 2021 "real pandemic dip" from the "web-mode shift" — they coincide
and GSS has no within-year mode experiment — so the web-mode term absorbs both. We validate the period
model by leave-one-year-out (does the national mood generalize, or is it overfit?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microhappiness.binning import PREDICTORS
from microhappiness.estimate import _FORMULA_RHS

WEB_WAVES = (2021, 2024)  # GSS push-to-web waves (COVID-era mode change)


def _prep(gss_binned):
    d = gss_binned.dropna(subset=["happy", *PREDICTORS]).copy()
    d["very_happy"] = (d["happy"] == 3).astype(int)
    d["web_mode"] = d["year"].isin(WEB_WAVES).astype(int)
    d["_w"] = d["wtssps"].fillna(1.0) if "wtssps" in d else 1.0
    return d


def _formula(df):
    return f"very_happy ~ {_FORMULA_RHS} + bs(year, df={df}) + web_mode"


def fit_temporal(gss_binned, *, df: int = 4):
    """Logit of HAPPY on composition + health + smooth period spline + web-mode flag."""
    import statsmodels.formula.api as smf

    return smf.logit(_formula(df), data=_prep(gss_binned)).fit(disp=0)


def _rate(g):
    return float(np.average((g["happy"] == 3).astype(float), weights=g["_w"]) * 100)


def per_year_rates(gss_binned, logit) -> pd.DataFrame:
    """Per-year actual vs modeled national very-happy rate (in-sample fit of the period model)."""
    d = _prep(gss_binned)
    d["pred"] = logit.predict(d) * 100
    rows = [{"year": int(y), "actual": _rate(g), "modeled": float(np.average(g["pred"], weights=g["_w"]))}
            for y, g in d.groupby("year")]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ----- Per-year COMPOSITIONAL panel ---------------------------------------------------------------
# A panel must avoid two traps: PLACES health doesn't exist pre-2020, and ACS income is binned in NOMINAL
# dollars (inflation alone would fake a rising-happiness trend). So the compositional panel uses only the
# inflation-immune demographic shares whose composition genuinely shifts over a decade. The result is the
# part of the temporal story we CAN model honestly — how an area's demographics moved its happiness —
# distinct from the national mood (unforecastable, see above) and the survey-mode artifact.
CIRC = ("married", "employment", "home_owner", "lives_alone")
CIRC_RHS = "married + C(employment) + home_owner + lives_alone"


def fit_circumstantial(gss_binned):
    """Fit the inflation-immune compositional model + seed (shared across all panel years)."""
    import statsmodels.formula.api as smf

    from microhappiness.poststratify import precompute_masks

    d = gss_binned.dropna(subset=["happy", *CIRC]).copy()
    d["very_happy"] = (d["happy"] == 3).astype(int)
    d["score"] = d["happy"].map({3: 100.0, 2: 50.0, 1: 0.0})
    logit = smf.logit(f"very_happy ~ {CIRC_RHS}", data=d).fit(disp=0)
    ols = smf.ols(f"score ~ {CIRC_RHS}", data=d).fit()
    d["_w"] = d["wtssps"].fillna(1.0) if "wtssps" in d else 1.0
    seed = d.groupby(list(CIRC), as_index=False)["_w"].sum()
    seed["_w"] /= seed["_w"].sum()
    seed["pct_very"] = logit.predict(seed) * 100
    seed["index"] = ols.predict(seed)
    masks = precompute_masks({p: seed[p].to_numpy() for p in CIRC})
    return masks, seed["_w"].to_numpy(), seed["index"].to_numpy(), seed["pct_very"].to_numpy()


def estimate_year(acs_margins, fitted) -> dict:
    """{geoid: compositional happiness_index} for one ACS vintage (rake only the CIRC margins)."""
    from microhappiness.poststratify import rake

    masks, w0, ix, _pv = fitted
    out = {}
    for geoid, margin in acs_margins.items():
        w = rake(masks, w0, {k: margin[k] for k in CIRC})
        out[geoid] = float(np.dot(w, ix))
    return out


def leave_one_year_out(gss_binned, *, df: int = 4) -> pd.DataFrame:
    """Refit excluding each year, predict that year's national rate out-of-sample (generalization test)."""
    import statsmodels.formula.api as smf

    d = _prep(gss_binned)
    rows = []
    for y in sorted(d["year"].unique()):
        tr, te = d[d["year"] != y], d[d["year"] == y]
        try:
            m = smf.logit(_formula(df), data=tr).fit(disp=0)
            pred = float(np.average(m.predict(te) * 100, weights=te["_w"]))
        except Exception:  # noqa: BLE001
            pred = np.nan
        rows.append({"year": int(y), "actual": _rate(te), "pred_oos": pred,
                     "web": int(y in WEB_WAVES)})
    return pd.DataFrame(rows)
