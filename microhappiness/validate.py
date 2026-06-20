"""Validate modeled estimates honestly — coarse geography is the strong axis, sub-city the weak one.

Three axes:
  1. County cross-section vs Sharecare Community Well-Being Index / Gallup (expect strong r at county,
     weaker sub-city — that's the documented MRP pattern, county r≈0.85-0.95 for health indicators).
  2. Tract proxy vs CDC PLACES poor-mental-health / poor-physical-health (adjacent, not identical).
  3. Longitudinal: synthetic national trend (aggregate all tracts per year) vs the ACTUAL GSS national
     HAPPY trend — an independent, free internal check the cross-section can't give.

Report correlations with honest framing; never imply tract estimates are validated as precise.
"""

from __future__ import annotations

BENCHMARKS = {
    "sharecare_cwbi": {"geo": "county/community", "note": "subscription; methods public"},
    "cdc_places_mental_health": {"geo": "tract/ZCTA", "note": "free; proxy, not happiness"},
    "gss_national_trend": {"geo": "national x year", "note": "free; from the GSS fit itself"},
}


def validate_county(estimates_tract, benchmark_county):
    """Aggregate tract estimates to county, correlate vs the benchmark, return r + scatter data."""
    raise NotImplementedError("tract->county pop-weighted, Pearson/Spearman vs benchmark")


def validate_national_trend(estimates_by_year, gss_national_by_year):
    """Compare the synthetic national happiness trend to the observed GSS national trend by year."""
    raise NotImplementedError("aggregate to nation per year; correlate vs GSS direct national means")
