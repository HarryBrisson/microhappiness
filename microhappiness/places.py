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
- CIRCULARITY: do NOT use PLACES 'mental health not good' / depression as a PREDICTOR — they are
  nearly the happiness construct itself. Those are VALIDATION targets only (validate.py).

Source: PLACES API (data.cdc.gov), tract + ZCTA releases.
"""

from __future__ import annotations

# Measures used as poststratification predictors (a strong, non-circular happiness correlate).
PREDICTOR_MEASURES = {
    "GHLTH": "fair-or-poor self-rated general health (% adults)",  # the health poststrat margin
}

# Measures reserved for VALIDATION only — too close to the happiness construct to use as predictors.
VALIDATION_MEASURES = {
    "MHLTH": "mental health not good >=14 days (% adults)",
    "DEPRESSION": "diagnosed depression (% adults)",
}

GEOGRAPHIES = ("tract", "zcta")


def fetch_health_marginal(geography: str, measure: str = "GHLTH", release: str | None = None):
    """{GEOID: fair_or_poor_health_fraction} for raking into the per-area joint table.

    Plan: pull the PLACES tract/ZCTA release from data.cdc.gov for `measure`, return the crude (or
    age-adjusted — pick one and document) prevalence as a fraction per GEOID, plus its CI so the
    synthetic-on-synthetic uncertainty can flow into estimate.monte_carlo_ci.
    """
    raise NotImplementedError("fetch PLACES measure -> {GEOID: fraction, ci}")
