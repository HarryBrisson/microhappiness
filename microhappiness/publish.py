"""Publish the nationwide tract/ZCTA estimates as a versioned artifact for downstream consumers.

Penlight (ward-wise) and collaborators consume the *published artifact*, not this repo's code — so
the modeling stack (statsmodels, big ACS pulls) never enters their app. We emit:

  estimates_tract_<vintage>.csv     GEOID, year, happiness_index, pct_very_happy, se, adult_pop
  estimates_zcta_<vintage>.csv
  aggregation_spec.json             byop/v1-style manifest describing the layers + metrics

The estimate is a per-area VALUE (not a point/raster), so a consumer allocates it to its own polygons
population-weighted (the same tract path ward-wise already uses for ACS/PLACES). The byop/v1 contract
gains a "weighted_mean" combine (value field x weight field) for this layer kind; until that lands,
ward-wise ingests the tract CSV directly on its existing tract->ward/CA/χGRID allocation, labeled
allocation_method="modeled_synthetic_sae", confidence low/medium.
"""

from __future__ import annotations

CONTRACT = "byop/v1"
SOURCE = "microhappiness"

METRICS = {
    "modeled_happiness_index": {
        "layer": "happiness_estimates",
        "combine": "weighted_mean",   # value=happiness_index, weight=adult_pop
        "value": "happiness_index",
        "weight": "adult_pop",
        "unit": "index_0_100",
        "direction": "higher_better",
        "synthetic": True,
        "label": "Modeled happiness",
    },
    "modeled_pct_very_happy": {
        "layer": "happiness_estimates",
        "combine": "weighted_mean",
        "value": "pct_very_happy",
        "weight": "adult_pop",
        "unit": "percent",
        "direction": "higher_better",
        "synthetic": True,
        "label": "Modeled % very happy",
    },
}


def write_artifact(out_dir, tract_rows, zcta_rows, *, vintage: str, model_key: str, gss_years: str):
    """Write the CSVs + aggregation_spec.json (with the mandatory synthetic-estimate caveat)."""
    raise NotImplementedError("write CSVs + spec; embed caveat, model_key, gss_years, acs vintage")
