"""Step 0 — the honesty gate.

Before building any small-area pipeline, answer the one question the literature could NOT pin down:
*how much of GSS happiness can ACS-derivable demographics actually explain?* We fit the fullest
ACS-derivable model (M0) on GSS microdata and report McFadden's pseudo-R² (plus a calibration
sketch). This is a go / no-go: if the ceiling is near-zero, the product is "demographically-implied
happiness with very low resolution" and we reframe (or stop) rather than ship false precision.

Run:  CENSUS_API_KEY unneeded here — GSS only.
      python -m diagnostics.step0_variance_ceiling --gss data/gss_cumulative.dta

This is intentionally dependency-light (pandas + statsmodels). It does NOT poststratify — it only
measures the model's ceiling on the survey itself.
"""

from __future__ import annotations

import argparse

from microhappiness.gss import load_gss, recode_predictors
from microhappiness.models import M0_CEILING


def mcfadden_r2(gss_path: str) -> dict:
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = recode_predictors(load_gss(gss_path, columns=None))
    df = df.dropna(subset=["happy", *M0_CEILING.cell_predictors])

    # Binary target first (very happy vs rest) — the most legible ceiling number.
    df["very_happy"] = (df["happy"] == 3).astype(int)
    terms = [f"C({p})" if p in ("marital", "race_ethnicity", "sex", "employment") else p
             for p in M0_CEILING.cell_predictors]
    if "age" in M0_CEILING.quadratic:
        terms += ["I(age**2)"]
    formula = "very_happy ~ " + " + ".join(terms)

    full = smf.glm(formula, data=df, family=sm.families.Binomial(),
                   freq_weights=df.get("wtssps")).fit()
    null = smf.glm("very_happy ~ 1", data=df, family=sm.families.Binomial(),
                   freq_weights=df.get("wtssps")).fit()
    r2 = 1.0 - (full.llf / null.llf)
    return {
        "n": int(df.shape[0]),
        "mcfadden_pseudo_r2": round(float(r2), 4),
        "base_rate_very_happy": round(float(df["very_happy"].mean()), 4),
        "pred_spread_p10_p90": [round(float(np.percentile(full.fittedvalues, q)), 4) for q in (10, 90)],
        "interpretation": _interpret(r2),
    }


def _interpret(r2: float) -> str:
    if r2 < 0.02:
        return "NEAR-ZERO ceiling — demographics barely move happiness. Reframe or stop; do not ship tract precision."
    if r2 < 0.06:
        return "LOW ceiling (expected). Estimates will be smooth/composition-driven — ship only with strong caveats."
    return "Above the expected demographic ceiling — double-check for leakage before celebrating."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gss", default="data/gss_cumulative.dta", help="GSS cumulative datafile (.dta/.sav)")
    args = ap.parse_args()
    result = mcfadden_r2(args.gss)
    print(f"M0 ceiling on GSS HAPPY (very-happy vs rest), N={result['n']}:")
    print(f"  McFadden pseudo-R²   = {result['mcfadden_pseudo_r2']}")
    print(f"  base rate very-happy = {result['base_rate_very_happy']}")
    print(f"  predicted P spread   = {result['pred_spread_p10_p90']} (p10..p90)")
    print(f"  -> {result['interpretation']}")


if __name__ == "__main__":
    main()
