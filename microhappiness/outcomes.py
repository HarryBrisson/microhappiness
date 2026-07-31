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
- area covariates (ecological bridges — fit on coarse respondent attachments, projected on each
  area's own value):
    * logdens (fear, socializing): linear log10-density, belt-anchored on the fit side (public GSS
      reveals only SRCBELT — see density.py), each tract's actual density on the predict side. This
      is what lets the estimate vary within a city, where every tract shares one belt. Validated:
      linear log-density recovers ~90-99% of the categorical belt fit.
    * logcong (attendance): CBP religious organizations per 1,000 adults (cbp.py), attached to
      respondents at region x belt cell means — the SUPPLY side of the behavior, and the route by
      which denominational geography (e.g. Utah) enters, which composition cannot see. Gate:
      R2 0.056 -> 0.060, holdout region x belt +0.59 -> +0.80, region x era +0.70 -> +0.87.

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
    # Ecological area covariates, fit on coarse respondent attachments and projected on each area's
    # own value: "logdens" (density.py belt anchors) and/or "logcong" (cbp.py congregations per 1k
    # adults, region x belt cell means). The fit zeroes the covariate design column, so the intercept
    # carries the covariate=0 baseline and the per-area offset is coef x the area's own value.
    area_covariates: tuple[str, ...] = ()
    index_column: str | None = None     # optional 0-100 index column (attendance only)
    index: object = None                # series -> 0-100 score


OUTCOMES: tuple[OutcomeSpec, ...] = (
    OutcomeSpec("attendance", "attend", "weekly_attendance_pct", "attends nearly weekly or more",
                identity=True, top=lambda s: s >= 6, area_covariates=("logcong",),
                index_column="attendance_index", index=lambda s: s / 8.0 * 100.0),
    OutcomeSpec("no_religion", "relig", "no_religion_pct", "no religious affiliation",
                identity=True, top=lambda s: s == 4),
    OutcomeSpec("god_certain", "god", "god_certain_pct", "knows God exists, no doubts",
                identity=True, top=lambda s: s == 6),
    OutcomeSpec("strong_affiliation", "reliten", "strong_affiliation_pct",
                "strong religious affiliation", identity=True, top=lambda s: s == 1),
    OutcomeSpec("fear_walking", "fear", "fear_walking_pct",
                "afraid to walk alone at night nearby", identity=True, standardize_identity=True,
                area_covariates=("logdens",), top=lambda s: s == 1),
    OutcomeSpec("financial_satisfaction", "satfin", "financial_satisfaction_pct",
                "satisfied with present finances", identity=False, top=lambda s: s == 1),
    OutcomeSpec("social_trust", "trust", "social_trust_pct", "most people can be trusted",
                identity=False, top=lambda s: s == 1),
    OutcomeSpec("weekly_friends", "socfrend", "weekly_friends_pct",
                "spends an evening with friends weekly or more", identity=False, area_covariates=("logdens",),
                top=lambda s: s <= 2),
    OutcomeSpec("weekly_neighbors", "socommun", "weekly_neighbors_pct",
                "spends an evening with neighbors weekly or more", identity=False, area_covariates=("logdens",),
                top=lambda s: s <= 2),
    OutcomeSpec("weekly_bar", "socbar", "weekly_bar_pct", "goes to a bar/tavern weekly or more",
                identity=False, area_covariates=("logdens",), top=lambda s: s <= 2),
    OutcomeSpec("life_exciting", "life", "life_exciting_pct", "finds life exciting",
                identity=False, top=lambda s: s == 1),
)

# Evaluated and REJECTED (kept so the diagnostics can re-measure): PRAY's ceiling sits at the
# near-zero kill line and its holdout ordering is weak. Do not add to OUTCOMES without new evidence.
# Also rejected, screened 2026-07 with subpop-conditioned fits (not re-measurable by the generic
# step-0, so recorded here): HAPMAR very-happy-marriage among married adults (R2 0.015 — below the
# 0.02 kill line; marriage quality isn't circumstantially structured) and SATJOB very-satisfied
# among workers (R2 0.015 AND region ordering -0.98/-0.74 — satisfied workers are where the
# composition least predicts them; both blades fail).
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


def fit_outcome(gss_binned, spec, seed, respondent_covs, *, rng_seed=0):
    """Fit one outcome, predict LINEAR predictors on the seed cells.

    Returns {column: {"link", "lin" (C,), "lin_draws" (C,D), "covs": {name: (coef, draws)}}}.
    `respondent_covs` maps each covariate name to a callable(frame) -> respondent values (the
    ecological attachment: logdens from belt anchors, logcong from region x belt cell means)."""
    import patsy
    import statsmodels.formula.api as smf

    dims = bundle_dims(spec.identity)
    rhs = IDENTITY_RHS if spec.identity else _FORMULA_RHS
    need = [spec.gss_col, *dims] + (["srcbelt"] if spec.area_covariates else [])
    d = gss_binned.dropna(subset=need).copy()
    for name in spec.area_covariates:
        rhs = rhs + f" + {name}"
        d[name] = respondent_covs[name](d)
        d = d.dropna(subset=[name])
    seed = seed.copy()
    for name in spec.area_covariates:
        seed[name] = 0.0  # placeholder column so the design matrix builds; offset added per area
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
        covs = {}
        for name in spec.area_covariates:
            j = names.index(name)
            covs[name] = (float(params.iloc[j]), b[:, j].copy())
            X[:, j] = 0.0  # the covariate contribution enters via the per-area offset instead
        return {"link": link, "lin": X @ params.to_numpy(), "lin_draws": X @ b.T, "covs": covs}

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


def _area_values(W, fit, deltas):
    """(values, lo, hi) per area for one column: value = W · link(lin + Σ coef·delta) per area.

    `deltas` maps covariate name -> per-area value vector; the fitted intercept carries the
    covariate=0 baseline (design columns zeroed in fit_outcome), so offsets use raw area values."""
    link = (lambda x: 100.0 / (1.0 + np.exp(-x))) if fit["link"] == "logit" else (lambda x: x)
    lin, draws, covs = fit["lin"], fit["lin_draws"], fit["covs"]
    if not covs:
        vals = W @ link(lin)
        lo, hi = np.percentile(W @ link(draws), [5, 95], axis=1)
        return vals, lo, hi
    offset = sum(coef * deltas[name] for name, (coef, _d) in covs.items())
    vals = np.einsum("tc,tc->t", W, link(lin[None, :] + offset[:, None]))
    per_draw = np.empty((W.shape[0], draws.shape[1]))
    for k in range(draws.shape[1]):  # loop draws: (T,C) at a time keeps memory bounded
        off_k = sum(dd[k] * deltas[name] for name, (_c, dd) in covs.items())
        per_draw[:, k] = np.einsum("tc,tc->t", W, link(draws[:, k][None, :] + off_k[:, None]))
    lo, hi = np.percentile(per_draw, [5, 95], axis=1)
    return vals, lo, hi


def estimate(gss_binned, acs_margins, places_health, area_covs, respondent_covs,
             *, chunk=2000, log=lambda s: None):
    """Estimate every outcome for every area with full margins + covariates. -> (DataFrame, meta).

    `area_covs`: {covariate name: {geoid: value}} (each area's own projection values);
    `respondent_covs`: {covariate name: callable(frame) -> values} (the fit-side attachment).
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
             if g in places_health and need_identity.issubset(m)
             and all(g in vals for vals in area_covs.values())]
    natl_identity = _national_identity_margins(areas)

    seeds = {ident: build_seed(gss_binned, bundle_dims(ident)) for ident in (True, False)}
    fits, n_fit = {}, {}
    for (ident, std), specs in groups.items():
        for spec in specs:
            covs = "+".join(spec.area_covariates)
            log(f"fitting {spec.key} ({'identity' if ident else 'circumstantial'}"
                f"{', standardized' if std else ''}{', ' + covs if covs else ''}) …")
            fits[spec.key], n_fit[spec.key] = fit_outcome(gss_binned, spec, seeds[ident],
                                                          respondent_covs)

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
            deltas = {name: np.array([vals[g] for g, _m, _gh in part])
                      for name, vals in area_covs.items()}
            for spec in specs:
                for col, fit in fits[spec.key].items():
                    vals, lo, hi = _area_values(W, fit, deltas)
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

    from microhappiness import calibrate, cbp, density
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
    print("building density + congregations layers (gazetteer + B01001 + CBP) …")
    # Anchors/reference always come from the national TRACT distribution (the belts describe tracts).
    ld_tract, pop_tract = density.log_density("tract", args.acs_year)
    anchors = density.belt_anchors(gss, ld_tract, pop_tract)
    log_dens = ld_tract if geo == "tract" else density.log_density(geo, args.acs_year)[0]
    ld_zcta, pop_zcta = density.log_density("zcta", args.acs_year)
    rate_zcta = cbp.log_congregation_rate("zcta", acs_year=args.acs_year)
    cong_cells = cbp.cell_rates(gss, rate_zcta, ld_zcta, pop_zcta, anchors)
    log_cong = rate_zcta if geo == "zcta" else cbp.log_congregation_rate("tract",
                                                                        acs_year=args.acs_year)
    # NO EXTRAPOLATION beyond the fitted supply range: the congregation coefficient is identified on
    # the 24 region x belt cell means, so area values are clipped to that span. This also defuses the
    # CBP zero-inflation artifact — half of (mostly rural, low-pop) ZCTAs show ZERO employer
    # congregations because volunteer-run congregations have no paid staff, which is measurement,
    # not absence; unclipped they'd sit ~0.6 log10 below the fitted support and crush attendance.
    lo_c, hi_c = min(cong_cells.values()), max(cong_cells.values())
    log_cong = {g: min(max(v, lo_c), hi_c) for g, v in log_cong.items()}
    ref_c = float(np.average([min(max(rate_zcta[z], lo_c), hi_c) for z in rate_zcta],
                             weights=[pop_zcta.get(z, 0.0) for z in rate_zcta]))
    for g in acs:  # areas with no CBP-linkable ZIP get the national mean supply (a neutral offset)
        log_cong.setdefault(g, ref_c)
    print(f"  belt anchors {dict(sorted((int(k), round(v, 2)) for k, v in anchors.items()))}; "
          f"congregation supply clipped to fitted range [{lo_c:.2f}, {hi_c:.2f}], ref {ref_c:.2f}")

    area_covs = {"logdens": log_dens, "logcong": log_cong}
    respondent_covs = {
        "logdens": lambda d: d["srcbelt"].map(anchors),
        "logcong": lambda d: pd.Series(
            [cong_cells.get((r, b)) for r, b in zip(d["region"], d["srcbelt"])], index=d.index,
            dtype="float"),
    }
    full, meta = estimate(gss, acs, places_health, area_covs, respondent_covs, log=print)
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
