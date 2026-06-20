"""Step 2 — fit the candidate models and measure the combined ceiling lift.

Two views:
  1. Each model on its OWN listwise sample (its achievable performance, with n shown — the PLACES
     predictors are asked only in some GSS years, so those samples shrink; that shrinkage is the point
     to see honestly).
  2. A COMMON-sample ladder: M3 (demographics) -> M4 (+health) -> M5 (+health+mental+smoking) all fit
     on the SAME rows (where the M5 predictors exist), so the incremental PLACES lift is apples-to-apples.

Pooled 5-fold out-of-sample McFadden pseudo-R² throughout (same metric as the screen). This answers:
how much do the PLACES margins lift the ceiling over the demographics-only ~0.041?

Run:  python -m diagnostics.step2_compare --gss data/gss_cumulative.dta
"""

from __future__ import annotations

from diagnostics.step1_screen import CATEGORICAL, _kfold_pseudo_r2
from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors
from microhappiness.models import M0_CEILING, M1_MINIMAL, M3_RICH, M4_HEALTH, M5_AFFECT


def formula_for(predictors, quadratic=()):
    terms = []
    for p in predictors:
        terms.append(f"C({p})" if p in CATEGORICAL else p)
        if p in quadratic:
            terms.append(f"I({p}**2)")
    return "very_happy ~ " + " + ".join(terms)


def fit_r2(df, predictors, quadratic=()):
    sub = df.dropna(subset=["very_happy", *predictors])
    if len(sub) < 300:
        return None, len(sub)
    return _kfold_pseudo_r2(formula_for(predictors, quadratic), sub), len(sub)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gss", default="data/gss_cumulative.dta")
    args = ap.parse_args()

    df = recode_predictors(load_gss(args.gss, columns=GSS_COLUMNS))
    df["very_happy"] = (df["happy"] == 3).astype(int)

    models = [M0_CEILING, M1_MINIMAL, M3_RICH, M4_HEALTH, M5_AFFECT]
    print("Each model on its own listwise sample (pooled 5-fold pseudo-R²):")
    print(f"  {'model':28} {'n':>7}  {'R²':>7}  predictors")
    for m in models:
        r2, n = fit_r2(df, m.cell_predictors, m.quadratic)
        print(f"  {m.label[:28]:28} {n:>7}  {('%.4f' % r2) if r2 is not None else '   n/a':>7}  "
              f"{len(m.cell_predictors)} vars")

    # The PLACES predictors live in different GSS year-modules, so a single all-complete sample is
    # empty. Measure each margin's lift over a CO-OBSERVED demographic core, each on its own sample
    # (core re-fit on the same rows for an apples-to-apples delta).
    core = ["marital", "income", "employment", "education", "home_owner", "lives_alone"]  # circumstantial
    print("\nPLACES-margin lift over the demographic core (each on its own co-observed sample):")
    print(f"  {'+ margin':28} {'n':>7}  {'core R²':>8}  {'+margin R²':>10}  {'lift':>7}")
    for label, extra in [("health", ["health"]), ("mental_health", ["mental_health"]),
                         ("smoker", ["smoker"]), ("all three", ["health", "mental_health", "smoker"])]:
        sub = df.dropna(subset=["very_happy", *core, *extra])
        if len(sub) < 300:
            print(f"  {label:28} {len(sub):>7}  (too few co-observed rows — different GSS modules)")
            continue
        base_r2 = _kfold_pseudo_r2(formula_for(core, ("age",)), sub)
        full_r2 = _kfold_pseudo_r2(formula_for(core + extra, ("age",)), sub)
        lift = f"+{full_r2 - base_r2:.4f}" if (base_r2 is not None and full_r2 is not None) else "n/a"
        print(f"  {label:28} {len(sub):>7}  {base_r2:>8.4f}  {full_r2:>10.4f}  {lift:>7}")


if __name__ == "__main__":
    main()
