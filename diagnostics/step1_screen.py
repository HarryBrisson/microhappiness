"""Step 1 — broad empirical screen on GSS's own variables, then a theory gate.

Discovery, NOT the final model. The exploratory pass can only run where individual happiness and the
candidate predictor sit together — that's GSS itself (hundreds of items), not ACS/PLACES (no fine
geography to attach to a GSS respondent). So we screen GSS-wide to see what actually predicts
happiness in-sample, then keep only variables that are simultaneously:
  (a) empirically useful (incremental, cross-validated signal — not noise),
  (b) reproducible at tract/ZCTA via ACS or PLACES, and
  (c) backed by a clear, logical mechanism (a HUMAN call — guards against spurious correlations).

This re-ranks predictors against the data instead of trusting the literature, while the theory gate
keeps the final model parsimonious (the research warned explicitly against kitchen-sink overfitting).

Guards against a broad screen chasing noise: a held-out split + cross-validated incremental
pseudo-R², and reporting effect stability across GSS eras (a real predictor is stable; a fluke isn't).
"""

from __future__ import annotations

# Local availability of a discovered construct — the (b) filter. Hand-maintained as the screen runs.
LOCAL_SOURCE = {
    "marital": "ACS B12001",
    "income": "ACS B19001",
    "employment": "ACS B23025",
    "education": "ACS B15003",
    "age": "ACS B01001",
    "sex": "ACS B01001",
    "race_ethnicity": "ACS B03002",
    "health": "CDC PLACES GHLTH (fair-or-poor)",   # the PLACES-unlocked margin
    "mental_health": "CDC PLACES MHLTH (some GSS years only)",
    "smoker": "CDC PLACES CSMOKING (some GSS years only)",
    "home_owner": "ACS B25003 (tenure)",           # GNH material wellbeing
    "lives_alone": "ACS B11001 (1-person households)",  # GNH social connectedness
    "num_children": "ACS B09002 / B11003",         # GNH material/social
    "hours_worked": "ACS B23022 (usual hours)",    # GNH time balance
    # constructs GSS shows matter but ACS/PLACES can't supply -> cannot poststratify (document, exclude):
    "attend": None,      # religious attendance
    "trust": None,       # social trust
    "polviews": None,    # political ideology
    "satfin": None,      # financial satisfaction (subjective; partly proxied by income)
}


CATEGORICAL = {"marital", "race_ethnicity", "sex", "employment"}


def _kfold_pseudo_r2(formula, data, *, folds: int = 5, seed: int = 0):
    """Pooled k-fold out-of-sample McFadden pseudo-R² — stable (no single-split noise).

    Each fold is scored on out-of-fold predictions; we pool the held-out log-likelihoods across all
    folds and compare to an intercept-only null, so the result is a single robust number.
    """
    import numpy as np
    import statsmodels.formula.api as smf

    y = data["very_happy"].to_numpy()
    fold_id = np.random.default_rng(seed).integers(0, folds, len(data))
    oos_ll = null_ll = 0.0
    for k in range(folds):
        tr, te = data[fold_id != k], data[fold_id == k]
        if len(te) == 0 or tr["very_happy"].nunique() < 2:
            continue
        try:
            p = np.clip(smf.logit(formula, data=tr).fit(disp=0).predict(te), 1e-6, 1 - 1e-6)
        except Exception:  # noqa: BLE001
            return None
        yt = te["very_happy"].to_numpy()
        base = min(max(float(tr["very_happy"].mean()), 1e-6), 1 - 1e-6)
        oos_ll += float(np.sum(yt * np.log(p) + (1 - yt) * np.log(1 - p)))
        null_ll += float(np.sum(yt * np.log(base) + (1 - yt) * np.log(1 - base)))
    return (1.0 - oos_ll / null_ll) if null_ll else None


def screen(gss_df, candidate_vars, *, folds: int = 5, seed: int = 0):
    """Rank candidate GSS variables by pooled k-fold out-of-sample signal for HAPPY (very-happy vs rest).

    Returns rows sorted by oos_pseudo_r2: {var, oos_pseudo_r2, n, era_sign_stable, local_source,
    mechanism_note}. A variable advances to a candidate model only if it has held-out signal AND a
    non-null local_source AND an accepted mechanism — empirical + reproducible + logical.
    """
    df = gss_df.copy()
    df["very_happy"] = (df["happy"] == 3).astype(int)
    rows = []
    for var in candidate_vars:
        if var not in df.columns:
            rows.append({"var": var, "oos_pseudo_r2": None, "n": 0, "note": "absent in this GSS file"})
            continue
        sub = df[["very_happy", var, "year"]].dropna()
        if sub[var].nunique() < 2 or len(sub) < 200:
            rows.append({"var": var, "oos_pseudo_r2": None, "n": int(len(sub)), "note": "too sparse"})
            continue
        rhs = f"C({var})" if var in CATEGORICAL else var
        r2 = _kfold_pseudo_r2(f"very_happy ~ {rhs}", sub, folds=folds, seed=seed)
        rows.append({
            "var": var,
            "oos_pseudo_r2": round(r2, 4) if r2 is not None else None,
            "n": int(len(sub)),
            "era_sign_stable": _era_stable(sub, var, rhs),
            "local_source": LOCAL_SOURCE.get(var, "??? (check ACS/PLACES availability)"),
            "mechanism_note": "",  # human fills the (c) gate
        })
    rows.sort(key=lambda r: (r.get("oos_pseudo_r2") is None, -(r.get("oos_pseudo_r2") or 0)))
    return rows


def _era_stable(sub, var, rhs):
    """True if the univariate coefficient keeps its sign across GSS decades (a real signal is stable)."""
    import statsmodels.formula.api as smf

    signs = set()
    for _, era in sub.assign(decade=(sub["year"] // 10 * 10)).groupby("decade"):
        if len(era) < 200 or era[var].nunique() < 2 or era["very_happy"].nunique() < 2:
            continue
        try:
            params = smf.logit(f"very_happy ~ {rhs}", data=era).fit(disp=0).params
            signs.add(bool(params.drop("Intercept", errors="ignore").mean() > 0))
        except Exception:  # noqa: BLE001
            continue
    return len(signs) <= 1


def main() -> None:
    import argparse

    from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors

    ap = argparse.ArgumentParser(description="Broad GSS predictor screen -> theory gate (Step 1).")
    ap.add_argument("--gss", default="data/gss_cumulative.dta")
    args = ap.parse_args()

    # ACS/PLACES-aligned predictors + a few non-ACS GSS items, to show the screen ranges wide.
    extra = ("attend", "polviews", "satfin")  # religious attendance, ideology, financial satisfaction
    df = recode_predictors(load_gss(args.gss, columns=GSS_COLUMNS + extra))
    for col, hi in (("attend", 8), ("polviews", 7), ("satfin", 3)):  # blank GSS DK/NA codes
        if col in df:
            df[col] = df[col].where(df[col] <= hi)

    candidates = ["marital", "income", "employment", "education", "age", "sex", "race_ethnicity",
                  "health", "mental_health", "home_owner", "lives_alone", "num_children",
                  "hours_worked", "attend", "polviews", "satfin"]
    rows = screen(df, candidates)
    print(f"{'predictor':15} {'oosR2':>7} {'n':>7} {'stable':>6}  local source / note")
    print("-" * 72)
    for r in rows:
        src = r.get("local_source") or r.get("note", "")
        print(f"{r['var']:15} {str(r.get('oos_pseudo_r2')):>7} {r.get('n', 0):>7} "
              f"{str(r.get('era_sign_stable', '')):>6}  {src}")


if __name__ == "__main__":
    main()
