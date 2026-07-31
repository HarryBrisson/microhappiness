"""Modeled GSS outcomes beyond happiness: religion, fear, financial satisfaction, trust, socializing.

Same MRP machinery as the happiness model (fit on GSS, predict on a poststrat seed, rake onto each
area's ACS + PLACES GHLTH margins), generalized to a SPEC-DRIVEN set of outcomes with three
per-outcome capabilities:

- identity (age4/sex/race_ethnicity in the fit). Per Harry: the identity-free equity policy protects
  WELLBEING metrics (a wellbeing map must not reward/penalize composition); DESCRIPTIVE metrics may
  use identity. Two modes:
    * composition (religion family): identity margins raked at TRACT values — the map deliberately
      reflects who lives there.
    * standardized (fear): identity is in the FIT (clean coefficients) but predictions hold the
      identity mix at the NATIONAL average — adjusts for response composition (women/older adults
      report more fear) without mapping it. Direct standardization, as in age-adjusted disease rates.
- density (fear, socializing): a linear log10-density term, fit on belt-anchored respondent density
  (public GSS reveals only SRCBELT — see density.py) and projected on each tract's ACTUAL density.
  This is what lets the estimate vary within a city, where every tract shares one belt. Validated:
  linear log-density recovers ~90-99% of the categorical belt fit.

GATE (diagnostics/step0_outcomes_ceiling.py): pseudo-R2 above the 0.02 kill line AND holdout
predictions that ORDER GSS geography (region / region x era / region x belt cells) — pseudo-R2 alone
is individual-level signal and can invert when projected onto places. The socializing trio fails
REGION ordering (cultural) but orders PLACE TYPES well once urbanicity is in (region x belt r
+0.37..+0.85) — published with caveats per Harry (2026-07), with weekly_bar carrying the strongest
caveat. Fear with identity+density is the best-modeled outcome here (pseudo-R2 ~0.11, region x belt
+0.96). PRAY remains REJECTED (0.024 at the kill line, holdout ~0.60).

National levels are calibrated to the RECENT (2018-2024) design-weighted GSS rates — attendance,
trust, and socializing decline secularly, so pooled benchmarks would overstate them.

Run (writes outcomes_<geo>.csv + merges the outcome metrics into aggregation_spec.json):
  python -m microhappiness.outcomes --geography tract --out-dir data/national
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microhappiness.binning import IDENTITY_PREDICTORS, PREDICTORS
from microhappiness.estimate import _FORMULA_RHS
from microhappiness.poststratify import precompute_masks, rake_many

IDENTITY_RHS = _FORMULA_RHS + " + C(age4) + C(sex) + C(race_ethnicity)"
CALIBRATION_YEARS = (2018, 2024)  # recent GSS waves: the level today, not the pooled-decades average
N_DRAWS = 200
SEED_SMOOTHING = 0.005  # share of seed mass spread uniformly over the full cell grid (see build_seed)


@dataclass(frozen=True)
class OutcomeSpec:
    key: str
    gss_col: str            # raw GSS source column (recoded/range-checked in gss.py)
    column: str             # published top-box share column, in percent
    label: str
    identity: bool          # age4/sex/race_ethnicity in the fit
    top: object             # series -> boolean top-box
    standardize_identity: bool = False  # predict at the NATIONAL identity mix (fear)
    density: bool = False               # + linear log-density (belt-anchored fit, tract projection)
    index_column: str | None = None     # optional 0-100 index column (attendance only)
    index: object = None                # series -> 0-100 score


OUTCOMES: tuple[OutcomeSpec, ...] = (
    OutcomeSpec("attendance", "attend", "weekly_attendance_pct", "attends nearly weekly or more",
                identity=True, top=lambda s: s >= 6,
                index_column="attendance_index", index=lambda s: s / 8.0 * 100.0),
    OutcomeSpec("no_religion", "relig", "no_religion_pct", "no religious affiliation",
                identity=True, top=lambda s: s == 4),
    OutcomeSpec("fear_walking", "fear", "fear_walking_pct",
                "afraid to walk alone at night nearby", identity=True, standardize_identity=True,
                density=True, top=lambda s: s == 1),
    OutcomeSpec("financial_satisfaction", "satfin", "financial_satisfaction_pct",
                "satisfied with present finances", identity=False, top=lambda s: s == 1),
    OutcomeSpec("social_trust", "trust", "social_trust_pct", "most people can be trusted",
                identity=False, top=lambda s: s == 1),
    OutcomeSpec("weekly_friends", "socfrend", "weekly_friends_pct",
                "spends an evening with friends weekly or more", identity=False, density=True,
                top=lambda s: s <= 2),
    OutcomeSpec("weekly_neighbors", "socommun", "weekly_neighbors_pct",
                "spends an evening with neighbors weekly or more", identity=False, density=True,
                top=lambda s: s <= 2),
    OutcomeSpec("weekly_bar", "socbar", "weekly_bar_pct", "goes to a bar/tavern weekly or more",
                identity=False, density=True, top=lambda s: s <= 2),
)

# Evaluated and REJECTED (kept so the diagnostics can re-measure): PRAY's ceiling sits at the
# near-zero kill line and its holdout ordering is weak. Do not add to OUTCOMES without new evidence.
REJECTED: tuple[OutcomeSpec, ...] = (
    OutcomeSpec("daily_prayer", "pray", "daily_prayer_pct", "prays daily or more",
                identity=False, top=lambda s: s <= 2),
)


def bundle_dims(identity: bool) -> tuple[str, ...]:
    return (*PREDICTORS, *IDENTITY_PREDICTORS) if identity else PREDICTORS


def build_seed(gss_binned, dims) -> pd.DataFrame:
    """Weighted GSS joint over `dims`, Laplace-smoothed onto the FULL level grid.

    The identity grid (192 x 4 x 2 x 4 = 6,144 cells) is sparse in ~20k GSS complete cases, and IPF
    cannot move mass into structurally-zero cells — so a small share (SEED_SMOOTHING) of the total
    weight is spread uniformly over every cell. Predictions are model-based (defined for all cells),
    so smoothing only completes the correlation structure; it cannot invent outcome signal."""
    d = gss_binned.dropna(subset=list(dims)).copy()
    d["_w"] = d["wtssps"].fillna(1.0) if "wtssps" in d else 1.0
    observed = d.groupby(list(dims), as_index=False)["_w"].sum()
    levels = [sorted(observed[p].unique()) for p in dims]
    full = pd.DataFrame(pd.MultiIndex.from_product(levels, names=list(dims)).to_frame(index=False))
    seed = full.merge(observed, on=list(dims), how="left").fillna({"_w": 0.0})
    seed["_w"] = (seed["_w"] * (1 - SEED_SMOOTHING) / seed["_w"].sum()
                  + SEED_SMOOTHING / len(seed))
    return seed


def fit_outcome(gss_binned, spec, seed, anchors, *, rng_seed=0):
    """Fit one outcome, predict LINEAR predictors on the seed cells.

    Returns {column: {"link", "lin" (C,), "lin_draws" (C,D), "dens" scalar, "dens_draws" (D,)}}.
    Density outcomes fit `+ logdens` on belt-anchored respondent density; `dens` is the coefficient
    the per-area projection applies to (tract logdens - national reference). Non-density outcomes
    carry dens=0, making the per-area math uniform."""
    import patsy
    import statsmodels.formula.api as smf

    dims = bundle_dims(spec.identity)
    rhs = IDENTITY_RHS if spec.identity else _FORMULA_RHS
    need = [spec.gss_col, *dims] + (["srcbelt"] if spec.density else [])
    d = gss_binned.dropna(subset=need).copy()
    if spec.density:
        rhs = rhs + " + logdens"
        d["logdens"] = d["srcbelt"].map(anchors)
    seed = seed.copy()
    seed["logdens"] = 0.0  # placeholder column so the design matrix builds; offset added per area
    rng = np.random.default_rng(rng_seed)
    out, n_fit = {}, len(d)

    def _fit(target_col, link):
        if link == "logit":
            model = smf.logit(f"{target_col} ~ {rhs}", data=d).fit(disp=0, maxiter=200)
        else:
            model = smf.ols(f"{target_col} ~ {rhs}", data=d).fit()
        params, cov = model.params, model.cov_params()
        X = np.asarray(patsy.build_design_matrices([model.model.data.design_info], seed)[0])
        b = rng.multivariate_normal(params.to_numpy(), cov.to_numpy(), N_DRAWS)
        names = list(params.index)
        if spec.density:
            j = names.index("logdens")
            dens, dens_draws = float(params.iloc[j]), b[:, j].copy()
            X[:, j] = 0.0  # logdens contribution enters via the per-area offset instead
        else:
            dens, dens_draws = 0.0, np.zeros(N_DRAWS)
        return {"link": link, "lin": X @ params.to_numpy(), "lin_draws": X @ b.T,
                "dens": dens, "dens_draws": dens_draws}

    d["_top"] = spec.top(d[spec.gss_col]).astype(int)
    out[spec.column] = _fit("_top", "logit")
    if spec.index_column:
        d["_idx"] = spec.index(d[spec.gss_col])
        out[spec.index_column] = _fit("_idx", "linear")
    return out, n_fit


def _national_identity_margins(areas) -> dict:
    """Adult-population-weighted national identity margins — the standardization targets (fear)."""
    out = {}
    for pred in IDENTITY_PREDICTORS:
        acc: dict = {}
        tot = 0.0
        for _geoid, m, gh in areas:
            w = gh["adult_pop"]
            tot += w
            for lvl, p in m[pred].items():
                acc[lvl] = acc.get(lvl, 0.0) + w * p
        out[pred] = {lvl: v / tot for lvl, v in acc.items()}
    return out


def _area_values(W, fit, delta):
    """(values, lo, hi) per area for one column: value = W · link(lin + dens*delta) per area."""
    link = (lambda x: 100.0 / (1.0 + np.exp(-x))) if fit["link"] == "logit" else (lambda x: x)
    lin, draws = fit["lin"], fit["lin_draws"]
    if fit["dens"] == 0.0 and not np.any(fit["dens_draws"]):
        vals = W @ link(lin)
        lo, hi = np.percentile(W @ link(draws), [5, 95], axis=1)
        return vals, lo, hi
    vals = np.einsum("tc,tc->t", W, link(lin[None, :] + fit["dens"] * delta[:, None]))
    per_draw = np.empty((W.shape[0], draws.shape[1]))
    for k in range(draws.shape[1]):  # loop draws: (T,C) at a time keeps memory bounded
        per_draw[:, k] = np.einsum(
            "tc,tc->t", W, link(draws[:, k][None, :] + fit["dens_draws"][k] * delta[:, None]))
    lo, hi = np.percentile(per_draw, [5, 95], axis=1)
    return vals, lo, hi


def estimate(gss_binned, acs_margins, places_health, log_dens, anchors,
             *, chunk=2000, log=lambda s: None):
    """Estimate every outcome for every area with full margins + density. -> (DataFrame, meta).

    Rake configurations (weights shared within each):
      identity-composition (religion): identity margins at TRACT values
      identity-standardized (fear):    identity margins at NATIONAL values
      circumstantial (the rest):       no identity margins
    """
    groups: dict = {}
    for spec in OUTCOMES:
        groups.setdefault((spec.identity, spec.standardize_identity), []).append(spec)

    need_identity = set(IDENTITY_PREDICTORS)
    areas = [(g, m, places_health[g]) for g, m in acs_margins.items()
             if g in places_health and need_identity.issubset(m) and g in log_dens]
    natl_identity = _national_identity_margins(areas)

    seeds = {ident: build_seed(gss_binned, bundle_dims(ident)) for ident in (True, False)}
    fits, n_fit = {}, {}
    for (ident, std), specs in groups.items():
        for spec in specs:
            log(f"fitting {spec.key} ({'identity' if ident else 'circumstantial'}"
                f"{', standardized' if std else ''}{', density' if spec.density else ''}) …")
            fits[spec.key], n_fit[spec.key] = fit_outcome(gss_binned, spec, seeds[ident], anchors)

    rows = {geoid: {"geoid": geoid, "adult_pop": gh["adult_pop"]} for geoid, _m, gh in areas}
    for (ident, std), specs in groups.items():
        dims = bundle_dims(ident)
        seed = seeds[ident]
        masks = precompute_masks({p: seed[p].to_numpy() for p in dims})
        w0 = seed["_w"].to_numpy()
        for lo_i in range(0, len(areas), chunk):
            part = areas[lo_i:lo_i + chunk]
            margin_list = []
            for _geoid, m, gh in part:
                mm = {p: m[p] for p in PREDICTORS if p != "health"}
                mm["health"] = {1.0: gh["fraction"], 0.0: 1.0 - gh["fraction"]}
                if ident:
                    src = natl_identity if std else m
                    for p in IDENTITY_PREDICTORS:
                        mm[p] = src[p]
                margin_list.append(mm)
            W = rake_many(masks, w0, margin_list, chunk=chunk)
            # The fitted intercept already carries the logdens=0 baseline (the design column is
            # zeroed in fit_outcome), so the per-area offset is dens_coef x the area's OWN logdens.
            delta = np.array([log_dens[g] for g, _m, _gh in part])
            for spec in specs:
                for col, fit in fits[spec.key].items():
                    vals, lo, hi = _area_values(W, fit, delta)
                    for i, (geoid, _m, _gh) in enumerate(part):
                        rows[geoid][col] = float(vals[i])
                        rows[geoid][f"{col}_lo"] = round(float(lo[i]), 2)
                        rows[geoid][f"{col}_hi"] = round(float(hi[i]), 2)
            log(f"  {'identity' if ident else 'circ'}{'/std' if std else ''} group: "
                f"{min(lo_i + chunk, len(areas))}/{len(areas)} areas")
    meta = {"n_fit": n_fit,
            "n_cells": {("identity" if k else "circ"): len(s) for k, s in seeds.items()}}
    return pd.DataFrame(list(rows.values())), meta


def gss_national_targets(gss_binned, years=CALIBRATION_YEARS) -> dict:
    """Design-weighted recent-window GSS national rate per published column — calibration targets."""
    out = {}
    for spec in OUTCOMES:
        d = gss_binned.dropna(subset=[spec.gss_col])
        d = d[d["year"].between(*years)]
        w = np.asarray(d["wtssps"].fillna(1.0) if "wtssps" in d else np.ones(len(d)), float)
        out[spec.column] = float(np.average(spec.top(d[spec.gss_col]).astype(float), weights=w) * 100)
        if spec.index_column:
            out[spec.index_column] = float(np.average(spec.index(d[spec.gss_col]), weights=w))
    return out


def main() -> None:
    import argparse
    from pathlib import Path

    from microhappiness import calibrate, density
    from microhappiness.acs import fetch_acs_margins_sf
    from microhappiness.binning import bin_gss
    from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors
    from microhappiness.places import fetch_measure
    from microhappiness.publish import write_outcomes_spec

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geography", choices=["tract", "zcta"], default="tract")
    ap.add_argument("--states", default="all", help="'all' or comma-separated FIPS (tract only)")
    ap.add_argument("--gss", default="data/gss_cumulative.dta")
    ap.add_argument("--acs-year", type=int, default=2022)
    ap.add_argument("--out-dir", default="data/national")
    args = ap.parse_args()
    geo = args.geography

    gss = bin_gss(recode_predictors(load_gss(args.gss, columns=GSS_COLUMNS)))
    print(f"fetching PLACES {geo} GHLTH margin …")
    places_health = fetch_measure("GHLTH", geography=geo)
    print(f"fetching ACS {geo} margins incl. identity (summary file) …")
    acs = fetch_acs_margins_sf(year=args.acs_year, geography=geo, include_identity=True)
    if geo == "tract" and args.states != "all":
        keep = set(args.states.split(","))
        acs = {g: m for g, m in acs.items() if g[:2] in keep}
    print(f"  {len(acs)} {geo}s with circumstantial margins")
    print("building density layer (gazetteer + B01001) …")
    # Anchors/reference always come from the national TRACT distribution (the belts describe tracts).
    ld_tract, pop_tract = density.log_density("tract", args.acs_year)
    anchors = density.belt_anchors(gss, ld_tract, pop_tract)
    log_dens = ld_tract if geo == "tract" else density.log_density(geo, args.acs_year)[0]
    print(f"  belt anchors {dict(sorted((int(k), round(v, 2)) for k, v in anchors.items()))}")

    full, meta = estimate(gss, acs, places_health, log_dens, anchors, log=print)
    target = gss_national_targets(gss)
    offs = calibrate.offsets(full, target)
    full = calibrate.apply_offsets(full, offs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"outcomes_{geo}.csv"
    full.to_csv(out, index=False)
    write_outcomes_spec(out_dir, [geo], vintage=str(args.acs_year), gss_years="1972-2024",
                        calibration=offs,
                        calibration_window=f"{CALIBRATION_YEARS[0]}-{CALIBRATION_YEARS[1]}")
    print(f"\nfits N={meta['n_fit']}; cells {meta['n_cells']}; {len(full)} {geo}s -> {out}")
    for spec in OUTCOMES:
        c = spec.column
        print(f"  {c}: median {full[c].median():.1f} (target {target[c]:.1f}, offset {offs[c]})")


if __name__ == "__main__":
    main()
