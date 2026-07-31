"""Step 0 for the modeled GSS outcomes — the honesty gate, rerun per outcome.

Same discipline as step0_variance_ceiling, with one addition that turned out to have teeth: besides
the pseudo-R2 ceiling, holdout predictions must ORDER GSS geography correctly (region and
region x era cells). Pseudo-R2 is individual-level signal; the socializing outcomes showed it can
INVERT when projected onto places (composition says young/unmarried/renting areas socialize more;
observed geographic/era variation runs the other way). The religion family (attendance, no_religion)
fits IDENTITY-AWARE (per Harry: descriptive metrics may use age/sex/race; wellbeing metrics may
not); the rest are circumstantial.

2026-07 verdicts (GSS cumulative through 2024) — published set in outcomes.OUTCOMES, failures in
outcomes.REJECTED. Second screen (same date): god_certain (R2 .065, region +.96) and
strong_affiliation (.044, +.96) join the identity religion family; life_exciting (.039, weak-positive
everywhere, no inversion) joins the caveated circumstantial set; HAPMAR (.015, below the kill line)
and SATJOB (.015 AND region -0.98) rejected — see the note above outcomes.REJECTED (their
subpop-conditioned fits aren't re-measurable by this generic script):
  PUBLISH: attendance* .055 (region r +.93) | no_religion* .081 (+.90) | financial_satisfaction
  .099 (recent-region +.90; pooled-region negative is an era-composition artifact — noted caveat) |
  social_trust .052 (+.98).                                    (* = identity-aware)
  REJECT: weekly_friends (region -0.62), weekly_neighbors (-0.30 recent, era -0.63), weekly_bar
  (equivocal +0.37..+0.61, era inverted), daily_prayer (.024 ceiling, holdout ~.60).

Run:  python -m diagnostics.step0_outcomes_ceiling --gss data/gss_cumulative.dta
"""

from __future__ import annotations

import argparse

import numpy as np


def outcome_ceilings(gss_path: str) -> dict:
    import statsmodels.formula.api as smf

    from microhappiness.binning import bin_gss
    from microhappiness.estimate import _FORMULA_RHS
    from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors
    from microhappiness.outcomes import IDENTITY_RHS, OUTCOMES, REJECTED, bundle_dims

    gss = bin_gss(recode_predictors(load_gss(gss_path, columns=GSS_COLUMNS)))
    out = {}
    for spec in (*OUTCOMES, *REJECTED):
        dims = bundle_dims(spec.identity)
        rhs = IDENTITY_RHS if spec.identity else _FORMULA_RHS
        need = [spec.gss_col, *dims]
        if spec.area_covariates:
            # The gate uses categorical SRCBELT — the information-equivalent of the production
            # log-density term (which needs tract data for its anchors).
            rhs = rhs + " + C(srcbelt)"
            need.append("srcbelt")
        d = gss.dropna(subset=need).copy()
        d["_top"] = spec.top(d[spec.gss_col]).astype(int)
        logit = smf.logit(f"_top ~ {rhs}", data=d).fit(disp=0, maxiter=200)
        null = smf.logit("_top ~ 1", data=d).fit(disp=0)

        # 5-fold holdout predictions, checked as predicted-vs-actual over region x era cells.
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(d))
        d["_pred"] = np.nan
        for k in range(5):
            m = smf.logit(f"_top ~ {rhs}", data=d[fold != k]).fit(disp=0, maxiter=200)
            d.loc[fold == k, "_pred"] = m.predict(d[fold == k])
        d["_era"] = (d["year"] // 12 * 12).astype(int)
        cells = d.groupby(["region", "_era"]).agg(actual=("_top", "mean"), pred=("_pred", "mean"),
                                                  n=("_top", "size"))
        cells = cells[cells["n"] >= 100]
        region = d.groupby("region").agg(actual=("_top", "mean"), pred=("_pred", "mean"))
        recent = d[d["year"] >= 2013].groupby("region").agg(actual=("_top", "mean"),
                                                            pred=("_pred", "mean"))
        belt_r = None
        if spec.area_covariates:
            bc = d.groupby(["region", "srcbelt"]).agg(actual=("_top", "mean"),
                                                      pred=("_pred", "mean"), n=("_top", "size"))
            bc = bc[bc["n"] >= 100]
            belt_r = round(float(bc["actual"].corr(bc["pred"])), 3)
        out[spec.key] = {
            "holdout_region_belt_r": belt_r,
            "identity": spec.identity,
            "published": spec in OUTCOMES,
            "n": int(len(d)),
            "base_rate": round(float(d["_top"].mean()), 4),
            "mcfadden_pseudo_r2": round(float(1.0 - logit.llf / null.llf), 4),
            "pred_spread_p10_p90": [round(float(np.percentile(logit.predict(d), q)), 4)
                                    for q in (10, 90)],
            "holdout_region_era_r": round(float(cells["actual"].corr(cells["pred"])), 3),
            "holdout_region_r": round(float(region["actual"].corr(region["pred"])), 3),
            "holdout_region_recent_r": round(float(recent["actual"].corr(recent["pred"])), 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gss", default="data/gss_cumulative.dta")
    args = ap.parse_args()
    for name, r in outcome_ceilings(args.gss).items():
        kind = "identity-aware" if r["identity"] else "circumstantial"
        verdict = "PUBLISHED" if r["published"] else "REJECTED"
        print(f"{name} ({kind}, {verdict}): N={r['n']}, base rate={r['base_rate']}")
        print(f"  McFadden pseudo-R2     = {r['mcfadden_pseudo_r2']}")
        print(f"  predicted P p10..p90   = {r['pred_spread_p10_p90']}")
        belt = f", region x belt {r['holdout_region_belt_r']}" if r["holdout_region_belt_r"] is not None else ""
        print(f"  holdout r: region x era {r['holdout_region_era_r']}, region "
              f"{r['holdout_region_r']}, region 2013+ {r['holdout_region_recent_r']}{belt}")


if __name__ == "__main__":
    main()
