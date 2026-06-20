"""Publish the estimates as a versioned artifact for downstream consumers (Penlight, collaborators).

Consumers read the *published artifact*, not this repo's code, so the modeling stack never enters their
app. We emit per-geography CSVs (geoid, happiness_index, pct_very_happy, adult_pop) + an
`aggregation_spec.json` (byop/v1-style) describing how to roll the per-area VALUES up to any polygons.

The estimate is a per-area value, so a consumer aggregates it population-weighted (the same tract path
ward-wise already uses for ACS/PLACES metrics), labeled allocation_method="modeled_synthetic_sae".
"""

from __future__ import annotations

import json
from pathlib import Path

CONTRACT = "byop/v1"
SOURCE = "microhappiness"
CAVEAT = ("Synthetic small-area estimates: the happiness EXPECTED given an area's circumstantial "
          "composition (income, marital/household status, employment, home ownership, health) — NOT an "
          "observed local measurement. Identity characteristics (age/sex/race) are excluded by policy.")

METRICS = {
    "modeled_happiness_index": {
        "combine": "weighted_mean", "value": "happiness_index", "weight": "adult_pop",
        "unit": "index_0_100", "direction": "higher_better", "synthetic": True,
        "label": "Modeled happiness", "category": "community_vitality",
    },
    "modeled_pct_very_happy": {
        "combine": "weighted_mean", "value": "pct_very_happy", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % very happy", "category": "community_vitality",
    },
}


def write_spec(out_dir, geographies, *, vintage: str, model_key: str, gss_years: str):
    """Write aggregation_spec.json next to the per-geography CSVs (happiness_<geo>.csv)."""
    out_dir = Path(out_dir)
    spec = {
        "contract": CONTRACT,
        "source": SOURCE,
        "synthetic_estimate": True,
        "caveat": CAVEAT,
        "acs_vintage": vintage,
        "model": model_key,
        "gss_years": gss_years,
        "layers": {
            geo: {"file": f"happiness_{geo}.csv", "kind": "polygon_values", "id_field": "geoid"}
            for geo in geographies
        },
        "metrics": {mid: {**m, "layer": list(geographies)} for mid, m in METRICS.items()},
    }
    (out_dir / "aggregation_spec.json").write_text(json.dumps(spec, indent=2))
    return out_dir / "aggregation_spec.json"
