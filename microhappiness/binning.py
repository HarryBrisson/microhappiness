"""Shared binning so the GSS fit, the poststrat seed, and the ACS/PLACES margins use IDENTICAL bins.

v1 poststratification predictors (chosen for clean ACS + PLACES margins; a documented simplification
of M4 — drops the weak `education`, bins income into 4 brackets):
  married (2) · employment (3: employed/unemployed/nilf) · home_owner (2) · lives_alone (2) ·
  income4 (4 brackets) · health (2: fair-or-poor vs not)  -> 2·3·2·2·4·2 = 192 cells.

UNIT NOTE (v1 approximation): marital/employment are person-level, tenure/income/household are
household-level, health is adult-level. v1 treats every margin as a proportion of the adult
population — an approximation (household size correlates with tenure/income); refine in v2.
"""

from __future__ import annotations

import numpy as np

PREDICTORS = ("married", "employment", "home_owner", "lives_alone", "income4", "health")
CATEGORICAL = ("employment",)  # the rest are already 0-indexed ints

# National household-income bracket percentile cut points (approx. US distribution) used to bin GSS
# income onto the same 4 brackets ACS publishes (<25k / 25-50k / 50-100k / 100k+).
INCOME_PCT_CUTS = (0.20, 0.43, 0.73)


def bin_gss(df):
    """Add the v1 binned predictor columns to an already-recoded GSS frame (see gss.recode_predictors)."""
    out = df.copy()
    out["married"] = (out["marital"] == "married").astype("float")
    # employment already {employed, unemployed, nilf}; home_owner / lives_alone / health already 0/1.
    pct = out.groupby("year")["realinc"].rank(pct=True)
    out["income4"] = np.digitize(pct, INCOME_PCT_CUTS).astype("float")
    out.loc[pct.isna(), "income4"] = np.nan
    for col in ("home_owner", "lives_alone", "health"):
        if col in out:
            out[col] = out[col].astype("float")
    return out
