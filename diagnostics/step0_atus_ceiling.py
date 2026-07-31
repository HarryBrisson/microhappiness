"""Step 0 for the ATUS time-use outcomes — the honesty gate, before anything is published.

Same two-blade discipline as step0_outcomes_ceiling (variance ceiling AND holdout geographic
ordering), with two upgrades ATUS makes possible:

- REAL sub-state geography for the ordering blade. ATUS public use carries state FIPS + metro
  status (GSS only ever revealed region + belt), so holdout predictions are checked against actual
  weighted outcome means over STATE cells and region x metro cells — a much sharper ordering test.
- An EXTERNAL anchor: the two-part commute model's holdout predictions are checked against the same
  cells' actual diary commutes here, and (in atus_outcomes.main) the tract-level modeled commute is
  correlated against ACS's independently MEASURED tract commute (B08013/B08303). GSS had no outcome
  with a censused ground truth; commute is exactly that for the time-use family.

All fits, cell means, and R2s are TUFNWGTP-weighted (each row is one diary day; the weight carries
the weekend oversampling correction).

Kill line: weighted R2 / pseudo-R2 >= 0.02 (repo convention) AND positive holdout ordering on the
state and region x metro axes.

Run:  python -m diagnostics.step0_atus_ceiling
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def _wmean(g: pd.DataFrame, col: str) -> float:
    return float(np.average(g[col], weights=g["tufnwgtp"]))


def _cell_ordering(d: pd.DataFrame, actual: str, pred: str, dims: list[str], min_n: int) -> dict:
    cells = (d.groupby(dims)
             .apply(lambda g: pd.Series({"actual": _wmean(g, actual), "pred": _wmean(g, pred),
                                         "n": len(g)}), include_groups=False))
    cells = cells[cells["n"] >= min_n]
    return {"r": round(float(cells["actual"].corr(cells["pred"])), 3),
            "spearman": round(float(cells["actual"].corr(cells["pred"], method="spearman")), 3),
            "n_cells": int(len(cells))}


def atus_ceilings() -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    from microhappiness.atus import ATUS_PREDICTORS, load_atus
    from microhappiness.atus_outcomes import ATUS_RHS, COMMUTE_ANY, COMMUTE_MIN, OUTCOMES

    d = load_atus().dropna(subset=list(ATUS_PREDICTORS)).copy()
    d["_w"] = d["tufnwgtp"] / d["tufnwgtp"].mean()
    rng = np.random.default_rng(0)
    d["_fold"] = rng.integers(0, 5, len(d))
    out = {}

    def _fit(frame, spec):
        if spec.kind == "minutes":
            return smf.wls(f"_y ~ {ATUS_RHS}", data=frame, weights=frame["_w"]).fit()
        return smf.glm(f"_y ~ {ATUS_RHS}", data=frame, freq_weights=frame["_w"],
                       family=sm.families.Binomial()).fit()

    for spec in (*OUTCOMES, COMMUTE_ANY, COMMUTE_MIN):
        f = d.copy()
        f["_y"] = spec.target(f)
        if spec is COMMUTE_MIN:
            f = f[f["commute_min"] > 0]
        model = _fit(f, spec)
        if spec.kind == "minutes":
            ceiling = float(model.rsquared)
        else:  # McFadden on the same weighted likelihood
            null = smf.glm("_y ~ 1", data=f, freq_weights=f["_w"],
                           family=sm.families.Binomial()).fit()
            ceiling = float(1.0 - model.llf / null.llf)
        f["_pred"] = np.nan
        for k in range(5):
            m = _fit(f[f["_fold"] != k], spec)
            f.loc[f["_fold"] == k, "_pred"] = m.predict(f[f["_fold"] == k])
        digits = 1 if spec.kind == "minutes" else 3
        out[spec.key] = {
            "n": int(len(f)),
            "weighted_ceiling_r2": round(ceiling, 4),
            "pred_spread_p10_p90": [round(float(np.percentile(f["_pred"], q)), digits)
                                    for q in (10, 90)],
            "holdout_state": _cell_ordering(f, "_y", "_pred", ["state_fips"], 500),
            "holdout_region_metro": _cell_ordering(f.dropna(subset=["metro"]), "_y", "_pred",
                                                   ["region", "metro"], 500),
            "published": spec.published,
        }
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name, r in atus_ceilings().items():
        verdict = "candidate" if r["published"] else "validation-only"
        print(f"{name} ({verdict}): N={r['n']}")
        print(f"  weighted ceiling R2    = {r['weighted_ceiling_r2']}  (kill line 0.02)")
        print(f"  predicted p10..p90     = {r['pred_spread_p10_p90']}")
        hs, hm = r["holdout_state"], r["holdout_region_metro"]
        print(f"  holdout ordering: state r {hs['r']} / rho {hs['spearman']} ({hs['n_cells']} cells), "
              f"region x metro r {hm['r']} / rho {hm['spearman']} ({hm['n_cells']} cells)")


if __name__ == "__main__":
    main()
