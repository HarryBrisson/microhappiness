"""Candidate happiness-model specifications.

Comprehensive ACS *data*, deliberately small *models*. Each spec names only ACS/PLACES-derivable
predictors so it can be poststratified onto every tract/ZCTA. We fit all of them, compare
McFadden pseudo-R² + calibration + cross-validated agreement against the benchmarks, and pick
per the evidence — we do not assume a winner. See METHODOLOGY.md.

EQUITY POLICY — circumstantial predictors only. We deliberately model happiness from MUTABLE,
circumstantial factors (income, marital/household status, employment, home ownership, health) and
NEVER from immutable identity characteristics (age, sex, race/ethnicity, nativity, veteran status).
Rationale: an area can't change its identity composition, and a wellbeing map must not "reward" or
"penalize" neighborhoods for who lives there — only for actionable conditions. Empirically this is
nearly free: dropping all identity predictors costs only ~0.004 in pseudo-R² (the identity variables
are weak), while the real levers (marital, income, health) are all circumstantial. See
`IMMUTABLE_IDENTITY` and METHODOLOGY.md §Equity.

A predictor name maps to a recode in `gss.py` (GSS side) and a poststrat dimension or area covariate
in `acs.py`/`places.py` (ACS/PLACES side); keep the two in lockstep so the join is valid.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    purpose: str
    # Individual-level predictors poststratified on the per-area joint table.
    cell_predictors: tuple[str, ...]
    # Area-level contextual covariates (PLACES-style) joined by GEOID, not poststratified.
    area_covariates: tuple[str, ...] = ()
    # Terms entered as nonlinear (e.g. "income" -> income + income**2 for diminishing returns).
    quadratic: tuple[str, ...] = ()
    # Grouping factors for random effects (e.g. region/state, year for the temporal overlay).
    random_effects: tuple[str, ...] = ()
    # Predictor x era interactions for the pooled-decades temporal fit.
    era_interactions: tuple[str, ...] = ()
    notes: str = ""


# Immutable identity characteristics — EXCLUDED from every production model by policy (not by fit).
# They are still reported in the transparency catalog (with this status) so the exclusion is visible.
IMMUTABLE_IDENTITY = ("age", "sex", "race_ethnicity", "us_born", "veteran")

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
    label="Circumstantial ceiling probe",
    purpose="Measure the variance ceiling of the fullest CIRCUMSTANTIAL set (the step-0 honesty gate). "
    "Identity-free by policy.",
    cell_predictors=("marital", "income", "employment", "education", "home_owner", "lives_alone"),
    notes="Not a production model per se — the diagnostic ceiling. If pseudo-R² is near-zero, reframe.",
)

M1_MINIMAL = ModelSpec(
    key="m1_minimal",
    label="Minimal / transparent",
    purpose="The two strongest circumstantial signals. Most interpretable.",
    cell_predictors=("marital", "income"),
)

M2_AREA = ModelSpec(
    key="m2_area",
    label="Area-covariate (circumstantial)",
    purpose="PLACES-style: a few individual circumstantial cells + area-level circumstantial covariates "
    "+ region random effect. Sidesteps full joint-table synthesis.",
    cell_predictors=("marital", "income", "employment"),
    area_covariates=("median_income_z", "pct_owner_occupied", "pct_fair_poor_health"),
    random_effects=("region",),
    notes="Area covariates are all circumstantial/actionable — never identity composition.",
)

M3_RICH = ModelSpec(
    key="m3_rich",
    label="Circumstantial-rich",
    purpose="Fullest defensible circumstantial individual model with raked per-area joint tables.",
    cell_predictors=("marital", "income", "employment", "education", "home_owner", "lives_alone"),
    quadratic=("income",),
    random_effects=("region",),
    notes="Requires IPF/raking to synthesize the per-tract joint table (see poststratify.py).",
)

M4_HEALTH = ModelSpec(
    key="m4_health",
    label="+ self-rated health (PLACES-unlocked)",
    purpose="M3 + self-rated health — the strongest reproducible non-demographic predictor, made local "
    "via PLACES. The single biggest lever (~+0.015 pseudo-R²).",
    # `health` is fit on GSS HEALTH (individual) and raked in per-area from the PLACES GHLTH marginal;
    # the GSS seed carries the circumstance x health correlation. See places.py / poststratify.py.
    cell_predictors=("marital", "income", "employment", "education", "home_owner", "lives_alone", "health"),
    quadratic=("income",),
    random_effects=("region",),
    notes="Caveats: synthetic-on-synthetic (PLACES is itself modeled); align GSS HEALTH to PLACES "
    "'fair-or-poor'. PLACES mental-health stays a validation target for M1-M4 (which exclude it).",
)

M5_AFFECT = ModelSpec(
    key="m5_affect",
    label="+ mental-health + smoking",
    purpose="M4 + mental-health-days + smoking. Mental health is a DISTINCT axis (illbeing/affect vs "
    "HAPPY's life-evaluation), not circular. Smoking is a behavioral MARKER that captures distress "
    "people won't self-report (they'll admit smoking before 'anxious') — we estimate levels, not "
    "causes, so a distinct reproducible signal earns its place. Both raked from PLACES (MHLTH, CSMOKING).",
    cell_predictors=("marital", "income", "employment", "education", "home_owner", "lives_alone",
                     "health", "mental_health", "smoker"),
    quadratic=("income",),
    random_effects=("region",),
    notes="GSS MNTLHLTH/SMOKE are asked only in some years -> those coefficients fit on a GSS subset, "
    "and the predictors live in non-overlapping modules, so the full fit needs multiple imputation / "
    "per-coefficient pooling, not listwise (see estimate.py). Smoking adds only ~+0.002 incremental — "
    "kept as a distinct level marker, not a difference-maker.",
)

# The per-area joint table is raked from a national SEED = GSS (or PUMS) microdata, which already
# carries happiness + all CIRCUMSTANTIAL predictors INCLUDING health at the individual level. ACS
# supplies the circumstantial margins; PLACES supplies the health margins. GSS is model and seed.
JOINT_SEED = "gss"  # PUMS lacks HEALTH, so GSS is needed for the health axis regardless

CANDIDATES: dict[str, ModelSpec] = {
    m.key: m for m in (M0_CEILING, M1_MINIMAL, M2_AREA, M3_RICH, M4_HEALTH, M5_AFFECT)
}
# All models are identity-free (IMMUTABLE_IDENTITY excluded). M1-M4 also exclude mental health (the
# cautious set, so PLACES MHLTH can independently validate them); M5 includes it.
