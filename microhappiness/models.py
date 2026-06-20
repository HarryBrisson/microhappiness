"""Candidate happiness-model specifications.

Comprehensive ACS *data*, deliberately small *models*. Each spec names only ACS-derivable
predictors so it can be poststratified onto every tract/ZCTA. We fit all of them, compare
McFadden pseudo-R² + calibration + cross-validated agreement against the benchmarks, and pick
per the evidence — we do not assume a winner. See METHODOLOGY.md.

A predictor name maps to a recode in `gss.py` (GSS side) and a poststrat dimension or area
covariate in `acs.py` (ACS side); keep the two in lockstep so the join is valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    purpose: str
    # Individual-level predictors poststratified on the per-area joint table.
    cell_predictors: tuple[str, ...]
    # Area-level contextual covariates (PLACES-style) joined by GEOID, not poststratified.
    area_covariates: tuple[str, ...] = ()
    # Terms entered as nonlinear: "age" -> age + age**2; "income" -> log/decile handled in recode.
    quadratic: tuple[str, ...] = ()
    # Grouping factors for random effects (e.g. region/state, year for the temporal overlay).
    random_effects: tuple[str, ...] = ()
    # Predictor x era interactions for the pooled-decades temporal fit.
    era_interactions: tuple[str, ...] = ()
    notes: str = ""


# Shared outcome + weighting conventions (applied to every spec).
OUTCOME = "happy"  # GSS HAPPY recoded: 3=very, 2=pretty, 1=not too (ordinal) / very-vs-rest (binary)
GSS_WEIGHT = "wtssps"  # composite GSS weight; fall back to wtssall for older cumulative files
INDEX_CODING = {3: 100.0, 2: 50.0, 1: 0.0}  # -> happiness_index (0-100) = E[score]

# Temporal overlay applied on top of any spec when fitting the pooled 1972-present series.
TEMPORAL_OVERLAY = dict(
    random_effects=("year",),
    era_interactions=("marital", "income"),
    note="Fit pooled, apply era-appropriate coefficients to the matching ACS vintage -> tract x year panel. "
    "Flag/down-weight the 2021 web/COVID GSS wave; keep age-period-cohort claims modest.",
)


M0_CEILING = ModelSpec(
    key="m0_ceiling",
    label="Ceiling probe",
    purpose="Measure the ACS-derivable variance ceiling (McFadden R²) before committing — honesty gate.",
    cell_predictors=("marital", "income", "employment", "education", "age", "sex", "race_ethnicity"),
    quadratic=("age",),
    notes="Not necessarily a production model. If pseudo-R² is near-zero, reframe the product.",
)

M1_MINIMAL = ModelSpec(
    key="m1_minimal",
    label="Minimal / transparent",
    purpose="Two strongest demographic signals + the age U-shape. Most interpretable.",
    cell_predictors=("marital", "income", "age"),
    quadratic=("age",),
)

M2_PLACES = ModelSpec(
    key="m2_places",
    label="PLACES-analog",
    purpose="Mirror CDC PLACES: compositional census cells + area covariates + region random effect.",
    cell_predictors=("age", "sex", "race_ethnicity", "education"),
    area_covariates=("pct_married", "median_income_z"),
    quadratic=("age",),
    random_effects=("region",),
    notes="Sidesteps joint-table synthesis: hard-to-cross predictors (marital, income) enter as context.",
)

M3_RICH = ModelSpec(
    key="m3_rich",
    label="Disciplined-rich / full MRP",
    purpose="Fullest defensible individual model with raked per-area joint poststrat tables.",
    cell_predictors=("marital", "income", "employment", "education", "age", "sex", "race_ethnicity"),
    quadratic=("age",),
    random_effects=("region",),
    notes="Requires IPF/raking to synthesize the per-tract joint table (see poststratify.py).",
)

M4_HEALTH = ModelSpec(
    key="m4_health",
    label="Health-poststratified (PLACES-unlocked)",
    purpose="M3 + self-rated health — the strongest happiness predictor ACS lacks, made local via PLACES.",
    # `health` is fit on GSS HEALTH (individual) and raked in per-area from the PLACES GHLTH marginal;
    # the GSS/PUMS seed carries the demographics x health correlation. See places.py / poststratify.py.
    cell_predictors=("marital", "income", "employment", "education", "age", "sex", "race_ethnicity", "health"),
    quadratic=("age",),
    random_effects=("region",),
    notes="Expected to lift the variance ceiling most. Caveats: synthetic-on-synthetic (PLACES is "
    "itself modeled); align GSS HEALTH to PLACES 'fair-or-poor'; never use PLACES mental-health as a "
    "predictor (circular) — that's a validation target only.",
)

# The per-area joint table is raked from a national SEED = GSS (or PUMS) microdata, which already
# carries happiness + all predictors INCLUDING health at the individual level. ACS supplies the
# demographic margins; PLACES supplies the health margin. So GSS is both the fitted model and the seed.
JOINT_SEED = "gss"  # alternative: "pums" (larger, but lacks HEALTH -> would need GSS for the health axis)

CANDIDATES: dict[str, ModelSpec] = {m.key: m for m in (M0_CEILING, M1_MINIMAL, M2_PLACES, M3_RICH, M4_HEALTH)}
