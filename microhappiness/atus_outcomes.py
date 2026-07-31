"""Modeled ATUS time-use outcomes: daily leisure time + time poverty, poststratified onto tracts.

Same MRP shape as the GSS outcomes (fit on survey microdata -> predict on a poststrat seed -> rake
onto each area's ACS margins), pointed at the American Time Use Survey (atus.py). Differences from
outcomes.py that matter:

- WEIGHTS EVERYWHERE. Each ATUS respondent is one diary day and the final weight TUFNWGTP corrects
  the deliberate weekend oversampling, so the fits (WLS / weighted GLM), the seed joint, and the
  calibration targets are all TUFNWGTP-weighted (GSS fits could get away with unweighted logits;
  an unweighted ATUS fit would average a fictitious 3.5-day weekend week).
- The predictor frame is circumstantial-only (identity-free policy, same as the happiness/wellbeing
  metrics) and adds `fulltime` (usually works 35+ hours) — the dominant time-budget predictor —
  raked to an ACS B23022 margin. Health is NOT in the frame (ATUS carries no GSS-style HEALTH item
  in the linked core), so unlike the GSS outcomes no PLACES margin is raked.
- No hurdle for leisure: the weighted zero-share of daily leisure is ~5% (see atus.weighted_stats),
  so a single weighted OLS on minutes is used for the published mean. Commute IS zero-inflated
  (~67% zero diary days) and uses a two-part model (P(any) x minutes|any) — but commute is
  validation-only, never published.

OUTCOMES
- leisure_minutes: average daily minutes of leisure, BLS "Leisure and sports" convention (ATUS major
  categories 12 + 13, sports/exercise INCLUDED — documented in atus.py, comparable to BLS releases).
- time_poverty_pct: share of adults with under TIME_POVERTY_MIN = 120 leisure minutes on a day.
  The 2h/day line is an absolute threshold in the spirit of the leisure-deprivation literature
  (Kalenkoski, Hamrick & Andrews 2011 define time poverty via 50%/60% of median discretionary time;
  our weighted median daily leisure is ~255 min, putting 50%-of-median at ~127 min — so the round
  absolute 120-minute line and the relative convention nearly coincide, and we prefer the absolute
  line for interpretability and stability across reruns).

VALIDATION ANCHOR (an external check GSS never allowed): the same machinery fits diary commute
minutes and predicts each tract's expected commute; ACS independently MEASURES tract commutes
(B08013/B08303). The tract-level correlation between modeled and measured commute is direct evidence
of how much real cross-tract differentiation composition-raking recovers — reported in the spec and
by diagnostics/step0_atus_ceiling.py, alongside the usual variance-ceiling and holdout gates.

Run (writes atus_<geo>.csv + merges metrics into aggregation_spec.json):
  python -m microhappiness.atus_outcomes --geography tract --out-dir data/national
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microhappiness.atus import ATUS_PREDICTORS
from microhappiness.poststratify import precompute_masks, rake_many

ATUS_RHS = "married + C(employment) + home_owner + lives_alone + income4 + I(income4**2) + fulltime"
TIME_POVERTY_MIN = 120  # minutes of daily leisure below which a diary day counts as time-poor
N_DRAWS = 200
SEED_SMOOTHING = 0.005  # as outcomes.build_seed; the 192-cell circumstantial grid is nearly dense


@dataclass(frozen=True)
class AtusOutcomeSpec:
    key: str
    column: str
    label: str
    kind: str                     # "minutes" (weighted OLS) | "binary" (weighted logit, percent)
    target: object                # frame -> target series
    published: bool = True


OUTCOMES: tuple[AtusOutcomeSpec, ...] = (
    AtusOutcomeSpec("leisure", "leisure_minutes", "average daily leisure minutes (BLS cats 12+13)",
                    "minutes", lambda d: d["leisure_min"]),
    AtusOutcomeSpec("time_poverty", "time_poverty_pct",
                    f"under {TIME_POVERTY_MIN} leisure minutes per day", "binary",
                    lambda d: (d["leisure_min"] < TIME_POVERTY_MIN).astype(int)),
)

# Validation-only two-part commute model (never published; the anchor against ACS-measured commutes).
COMMUTE_ANY = AtusOutcomeSpec("commute_any", "commute_any_pct", "any work travel on the diary day",
                              "binary", lambda d: (d["commute_min"] > 0).astype(int), published=False)
COMMUTE_MIN = AtusOutcomeSpec("commute_pos", "commute_pos_min", "commute minutes among commuting days",
                              "minutes", lambda d: d["commute_min"], published=False)


def build_seed(atus: pd.DataFrame) -> pd.DataFrame:
    """TUFNWGTP-weighted ATUS joint over the predictor frame, Laplace-smoothed onto the full grid.

    Structurally-impossible cells (fulltime=1 while not employed) are zeroed AFTER smoothing so IPF
    can never grow mass in combinations no adult can occupy."""
    d = atus.dropna(subset=list(ATUS_PREDICTORS)).copy()
    d["_w"] = d["tufnwgtp"].astype(float)
    observed = d.groupby(list(ATUS_PREDICTORS), as_index=False)["_w"].sum()
    levels = [sorted(observed[p].unique()) for p in ATUS_PREDICTORS]
    full = pd.DataFrame(pd.MultiIndex.from_product(levels, names=list(ATUS_PREDICTORS))
                        .to_frame(index=False))
    seed = full.merge(observed, on=list(ATUS_PREDICTORS), how="left").fillna({"_w": 0.0})
    seed["_w"] = (seed["_w"] * (1 - SEED_SMOOTHING) / seed["_w"].sum() + SEED_SMOOTHING / len(seed))
    seed.loc[(seed["fulltime"] == 1.0) & (seed["employment"] != "employed"), "_w"] = 0.0
    seed["_w"] /= seed["_w"].sum()
    return seed


def fit_outcome(atus: pd.DataFrame, spec: AtusOutcomeSpec, seed: pd.DataFrame, *, rng_seed=0):
    """Weighted fit -> per-seed-cell linear predictors (+ coefficient draws). -> (fit dict, n)."""
    import patsy
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = atus.dropna(subset=[*ATUS_PREDICTORS]).copy()
    d["_y"] = spec.target(d)
    if spec is COMMUTE_MIN:  # the two-part mu: minutes among days with any commute
        d = d[d["commute_min"] > 0]
    d = d.dropna(subset=["_y"])
    w = d["tufnwgtp"].to_numpy(float)
    d["_w_norm"] = w / w.mean()  # mean-1 normalization: weighted point estimates, honest-N SEs
    if spec.kind == "minutes":
        model = smf.wls(f"_y ~ {ATUS_RHS}", data=d, weights=d["_w_norm"]).fit()
        link = "linear"
    else:
        model = smf.glm(f"_y ~ {ATUS_RHS}", data=d, freq_weights=d["_w_norm"],
                        family=sm.families.Binomial()).fit()
        link = "logit"
    X = np.asarray(patsy.build_design_matrices([model.model.data.design_info], seed)[0])
    rng = np.random.default_rng(rng_seed)
    b = rng.multivariate_normal(model.params.to_numpy(), model.cov_params().to_numpy(), N_DRAWS)
    return {"link": link, "lin": X @ model.params.to_numpy(), "lin_draws": X @ b.T}, len(d)


def _link(fit):
    return (lambda x: 100.0 / (1.0 + np.exp(-x))) if fit["link"] == "logit" else (lambda x: x)


def clamp_fulltime(margin: dict, ft_share: float) -> dict:
    """Add the fulltime margin, clamped under the employed share so the IPF targets stay coherent
    (the B23022 20-64 universe can nominally exceed a tract's 16+ employed share)."""
    employed = margin["employment"].get("employed", 0.0)
    ft = min(ft_share, 0.98 * employed)
    return {**margin, "fulltime": {1.0: ft, 0.0: 1.0 - ft}}


def estimate(atus, acs_margins, extras, *, chunk=2000, log=lambda s: None):
    """Estimate every published outcome + the commute anchor for every area. -> (DataFrame, meta)."""
    seed = build_seed(atus)
    fits, n_fit = {}, {}
    for spec in (*OUTCOMES, COMMUTE_ANY, COMMUTE_MIN):
        log(f"fitting {spec.key} ({spec.kind}, weighted) …")
        fits[spec.key], n_fit[spec.key] = fit_outcome(atus, spec, seed)

    areas = [(g, m, extras[g]) for g, m in acs_margins.items() if g in extras]
    masks = precompute_masks({p: seed[p].to_numpy() for p in ATUS_PREDICTORS})
    w0 = seed["_w"].to_numpy()
    rows = {}
    for lo_i in range(0, len(areas), chunk):
        part = areas[lo_i:lo_i + chunk]
        margin_list = [clamp_fulltime({p: m[p] for p in ATUS_PREDICTORS if p != "fulltime"},
                                      ex["fulltime_share"]) for _g, m, ex in part]
        W = rake_many(masks, w0, margin_list, chunk=chunk)
        cols = {}
        for spec in OUTCOMES:
            f = fits[spec.key]
            cols[spec.column] = W @ _link(f)(f["lin"])
            lo, hi = np.percentile(W @ _link(f)(f["lin_draws"]), [5, 95], axis=1)
            cols[f"{spec.column}_lo"], cols[f"{spec.column}_hi"] = lo, hi
        # Two-part commute anchor: per-adult expected minutes and per-commuting-day minutes.
        p = _link(fits["commute_any"])(fits["commute_any"]["lin"]) / 100.0
        mu = fits["commute_pos"]["lin"]
        cols["commute_expected_min"] = W @ (p * mu)
        cols["commute_conditional_min"] = (W @ (p * mu)) / np.maximum(W @ p, 1e-9)
        for i, (geoid, _m, ex) in enumerate(part):
            rows[geoid] = {"geoid": geoid, "adult_pop": ex["pop"],
                           **{c: round(float(v[i]), 2) for c, v in cols.items()}}
        log(f"  {min(lo_i + chunk, len(areas))}/{len(areas)} areas")
    return pd.DataFrame(list(rows.values())), {"n_fit": n_fit, "n_cells": len(seed)}


def atus_national_targets(atus) -> dict:
    """TUFNWGTP-weighted national rates for the published columns — the calibration targets."""
    out = {}
    for spec in OUTCOMES:
        d = atus.dropna(subset=list(ATUS_PREDICTORS)).copy()
        y = spec.target(d).to_numpy(float)
        w = d["tufnwgtp"].to_numpy(float)
        scale = 100.0 if spec.kind == "binary" else 1.0
        out[spec.column] = float(np.average(y, weights=w) * scale)
    return out


def commute_anchor(df: pd.DataFrame, extras: dict) -> dict:
    """Correlate modeled tract commutes against ACS-measured mean commutes — the validity anchor.

    The conditional (per-commuting-day) series is the apples-to-apples one: ACS mean commute is per
    commuting worker, so the expected-minutes series also folds in the employment margin."""
    acs = df["geoid"].map(lambda g: extras[g]["commute_mean"] if g in extras else np.nan)
    out = {}
    for col in ("commute_conditional_min", "commute_expected_min"):
        both = pd.DataFrame({"model": df[col], "acs": acs}).dropna()
        out[col] = {"pearson_r": round(float(both["model"].corr(both["acs"])), 3),
                    "spearman_r": round(float(both["model"].corr(both["acs"], method="spearman")), 3),
                    "n_tracts": int(len(both))}
    return out


def spread(df: pd.DataFrame, col: str) -> dict:
    """Population-weighted p10/p50/p90 of a published column — the is-the-map-flat check."""
    d = df.dropna(subset=[col])
    order = np.argsort(d[col].to_numpy())
    cum = np.cumsum(d["adult_pop"].to_numpy(float)[order])
    q = lambda f: float(d[col].to_numpy()[order][np.searchsorted(cum, f * cum[-1])])
    return {"p10": round(q(0.10), 1), "p50": round(q(0.50), 1), "p90": round(q(0.90), 1)}


def main() -> None:
    import argparse
    from pathlib import Path

    from microhappiness import calibrate
    from microhappiness.acs import fetch_acs_margins_sf, fetch_atus_extras_sf
    from microhappiness.atus import load_atus, weighted_stats
    from microhappiness.publish import write_atus_spec

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geography", choices=["tract", "zcta"], default="tract")
    ap.add_argument("--acs-year", type=int, default=2022)
    ap.add_argument("--out-dir", default="data/national")
    args = ap.parse_args()
    geo = args.geography

    atus = load_atus()
    yr_lo, yr_hi = int(atus["year"].min()), int(atus["year"].max())
    stats = weighted_stats(atus, "leisure_min")
    print(f"ATUS {yr_lo}-{yr_hi}: {len(atus)} adult diary days; weighted daily leisure "
          f"mean {stats['mean']:.0f} min, median {stats['median']:.0f}, zero share "
          f"{stats['zero_share']:.1%} (single-part OLS)")
    print(f"fetching ACS {geo} margins (summary file) …")
    acs = fetch_acs_margins_sf(year=args.acs_year, geography=geo)
    extras = fetch_atus_extras_sf(year=args.acs_year, geography=geo)
    print(f"  {len(acs)} {geo}s with circumstantial margins; {len(extras)} with fulltime/commute")

    full, meta = estimate(atus, acs, extras, log=print)
    target = atus_national_targets(atus)
    offs = calibrate.offsets(full, target)
    full = calibrate.apply_offsets(full, offs)

    anchor = commute_anchor(full, extras)
    spreads = {spec.column: spread(full, spec.column) for spec in OUTCOMES}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["geoid", "adult_pop"] + [c for spec in OUTCOMES
                                     for c in (spec.column, f"{spec.column}_lo", f"{spec.column}_hi")]
    out = out_dir / f"atus_{geo}.csv"
    full[keep].to_csv(out, index=False)
    write_atus_spec(out_dir, [geo], vintage=str(args.acs_year), atus_years=f"{yr_lo}-{yr_hi}",
                    calibration=offs, validation={"commute_anchor": anchor, "spread": spreads})
    print(f"\nfits N={meta['n_fit']}; cells {meta['n_cells']}; {len(full)} {geo}s -> {out}")
    for spec in OUTCOMES:
        print(f"  {spec.column}: target {target[spec.column]:.1f}, offset {offs[spec.column]}, "
              f"weighted p10/p50/p90 {spreads[spec.column]}")
    for col, r in anchor.items():
        print(f"  anchor {col}: pearson {r['pearson_r']}, spearman {r['spearman_r']} "
              f"over {r['n_tracts']} areas")


if __name__ == "__main__":
    main()
