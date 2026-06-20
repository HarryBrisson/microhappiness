"""CDC PLACES tract/ZCTA health estimates — the poststratification marginal ACS can't give us.

PLACES publishes modeled small-area health measures at tract AND ZCTA level (our exact target geo).
Its value here is specific, not generic: self-rated GENERAL health (% fair-or-poor) is one of the
strongest happiness predictors that ACS lacks. GSS carries self-rated health (HEALTH) at the
individual level, so we can FIT happiness ~ health; PLACES then supplies each area's health MARGINAL
to poststratify on, raked into the per-area joint table (the GSS/PUMS seed carries the
demographics x health correlation, so we never need an observed joint).

So: ACS = demographic margins, PLACES = the health margin, GSS = the model AND the joint seed.

CAUTIONS (see METHODOLOGY.md):
- Synthetic-on-synthetic: PLACES is itself modeled (BRFSS -> ACS MRP). Uncertainty compounds; label it.
- Definitional alignment: collapse GSS HEALTH (excellent/good/fair/poor) to PLACES 'fair-or-poor'
  so the raking control total is consistent with the fitted predictor.
- NOT circular: PLACES 'mental health not good' is an illbeing/affect axis, conceptually DISTINCT from
  HAPPY's global life-evaluation (wellbeing science separates life-evaluation, affect, eudaimonia,
  illbeing). So MHLTH is a legitimate predictor (model M5) AND an independent validation target for
  the models that EXCLUDE it (M1-M4). It just needs a GSS individual analog (MNTLHLTH) to fit on, which
  exists only in some GSS years. Depression has no clean GSS individual analog -> validation only.

Source: PLACES API (data.cdc.gov), tract + ZCTA releases.
"""

from __future__ import annotations

# Measures usable as poststratification predictors (have a GSS individual analog to fit on).
PREDICTOR_MEASURES = {
    "GHLTH": "fair-or-poor self-rated general health (% adults)",   # <- GSS HEALTH; the health margin
    "MHLTH": "mental health not good >=14 days (% adults)",         # <- GSS MNTLHLTH (subset of years); M5
    "CSMOKING": "current smoking (% adults)",                       # <- GSS SMOKE; behavioral distress marker; M5
}

# Measures used for VALIDATION (and as predictors only where a GSS analog exists, above).
VALIDATION_MEASURES = {
    "MHLTH": "mental health not good >=14 days (% adults)",   # validates M1-M4 (which exclude it)
    "DEPRESSION": "diagnosed depression (% adults)",          # no GSS analog -> validation only
}

GEOGRAPHIES = ("tract", "zcta")


def fetch_health_marginal(geography: str, measure: str = "GHLTH", release: str | None = None):
    """{GEOID: fair_or_poor_health_fraction} for raking into the per-area joint table.

    Plan: pull the PLACES tract/ZCTA release from data.cdc.gov for `measure`, return the crude (or
    age-adjusted — pick one and document) prevalence as a fraction per GEOID, plus its CI so the
    synthetic-on-synthetic uncertainty can flow into estimate.monte_carlo_ci.
    """
    raise NotImplementedError("fetch PLACES measure -> {GEOID: fraction, ci}")
