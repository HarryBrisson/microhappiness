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


RELIGION_CAVEAT = (
    "Synthetic small-area estimates: the religious outcome EXPECTED given an area's demographic AND "
    "circumstantial composition (age/sex/race plus income, marital/household status, employment, home "
    "ownership, health) — NOT an observed local measurement or a count of congregations. Unlike the "
    "wellbeing metrics, the religion metrics deliberately include identity: attendance and affiliation "
    "are strongly age- and race-patterned, and these are descriptive metrics, not wellbeing scores.")

CIRC_CAVEAT = (
    "Synthetic small-area estimates: the outcome EXPECTED given an area's circumstantial composition "
    "(income, marital/household status, employment, home ownership, health) — NOT an observed local "
    "measurement. Identity characteristics (age/sex/race) are excluded by policy, as in the happiness "
    "metrics.")

OUTCOME_METRICS = {
    "modeled_attendance_index": {
        "combine": "weighted_mean", "value": "attendance_index", "weight": "adult_pop",
        "unit": "index_0_100", "direction": "higher_better", "synthetic": True,
        "label": "Modeled religious attendance", "category": "religion_spiritual",
        "caveat": RELIGION_CAVEAT + " Also uses local congregation supply (CBP religious "
                  "organizations per capita), which carries denominational geography composition "
                  "alone cannot see — but inverts where congregations are few and large "
                  "(e.g. LDS Utah models far too low); reliable where congregations are "
                  "many and small, as in Chicago.",
    },
    "modeled_weekly_attendance_pct": {
        "combine": "weighted_mean", "value": "weekly_attendance_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % weekly attenders", "category": "religion_spiritual",
        "caveat": RELIGION_CAVEAT + " Also uses local congregation supply (CBP religious "
                  "organizations per capita), which carries denominational geography composition "
                  "alone cannot see — but inverts where congregations are few and large "
                  "(e.g. LDS Utah models far too low); reliable where congregations are "
                  "many and small, as in Chicago.",
    },
    "modeled_no_religion_pct": {
        "combine": "weighted_mean", "value": "no_religion_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "lower_better", "synthetic": True,
        "label": "Modeled % no religious affiliation", "category": "religion_spiritual",
        "caveat": RELIGION_CAVEAT + " Regional cultural secularism (e.g. the West's) is not "
                  "compositional and is largely missed; within-metro contrasts are the supported use.",
    },
    "modeled_god_certain_pct": {
        "combine": "weighted_mean", "value": "god_certain_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % certain God exists", "category": "religion_spiritual",
        "caveat": RELIGION_CAVEAT,
    },
    "modeled_strong_affiliation_pct": {
        "combine": "weighted_mean", "value": "strong_affiliation_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % strongly religious", "category": "religion_spiritual",
        "caveat": RELIGION_CAVEAT,
    },
    "modeled_life_exciting_pct": {
        "combine": "weighted_mean", "value": "life_exciting_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % finding life exciting", "category": "psychological_wellbeing",
        "caveat": CIRC_CAVEAT + " Geographic ordering is weak-positive on every axis (region "
                  "+0.2..+0.6) — treat as a soft signal, strongest for within-metro contrasts.",
    },
    "modeled_financial_satisfaction_pct": {
        "combine": "weighted_mean", "value": "financial_satisfaction_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % satisfied with finances", "category": "material_wellbeing",
        "caveat": CIRC_CAVEAT + " Cost of living is not modeled, so cross-metro comparisons "
                  "overstate satisfaction in expensive areas; within-metro contrasts are the "
                  "supported use.",
    },
    "modeled_social_trust_pct": {
        "combine": "weighted_mean", "value": "social_trust_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled social trust", "category": "community_vitality",
        "caveat": CIRC_CAVEAT,
    },
    "modeled_fear_walking_pct": {
        "combine": "weighted_mean", "value": "fear_walking_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "lower_better", "synthetic": True,
        "label": "Modeled % afraid to walk at night", "category": "community_vitality",
        "caveat": ("Synthetic small-area estimate of PERCEIVED safety: the share expected to report "
                   "being afraid to walk alone at night nearby, given the area's circumstantial "
                   "composition and population density. Identity (age/sex/race) is fitted but "
                   "STANDARDIZED to the national mix, so the map reflects conditions, not who lives "
                   "there. A perception measure — local crime itself is not an input."),
    },
}

_SOC_CAVEAT = (CIRC_CAVEAT + " Geographic validity is partial: the model orders place types "
               "(urban/suburban/rural, via the density term) but NOT regions — regional socializing "
               "differences are cultural, not compositional. Within-metro contrasts are the "
               "supported use.")
OUTCOME_METRICS.update({
    "modeled_weekly_friends_pct": {
        "combine": "weighted_mean", "value": "weekly_friends_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % seeing friends weekly", "category": "social_connectedness",
        "caveat": _SOC_CAVEAT,
    },
    "modeled_weekly_neighbors_pct": {
        "combine": "weighted_mean", "value": "weekly_neighbors_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % socializing with neighbors weekly", "category": "social_connectedness",
        "caveat": _SOC_CAVEAT,
    },
    "modeled_weekly_bar_pct": {
        "combine": "weighted_mean", "value": "weekly_bar_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "higher_better", "synthetic": True,
        "label": "Modeled % at a bar weekly", "category": "social_connectedness",
        "caveat": _SOC_CAVEAT + " The weakest of the socializing metrics: its geographic signal is "
                  "almost entirely the density gradient.",
    },
})
# PRAY remains evaluated-and-REJECTED (outcomes.REJECTED): ceiling at the kill line, weak holdout.


ATUS_CAVEAT = (
    "Synthetic small-area estimates from the American Time Use Survey: the time use EXPECTED given "
    "an area's circumstantial composition (employment and full-time work, income, marital/household "
    "status, home ownership) — NOT an observed local time diary. Identity characteristics "
    "(age/sex/race) are excluded by policy, as in the happiness metrics. Leisure follows the BLS "
    "'leisure and sports' convention (socializing/relaxing/leisure plus sports/exercise). "
    "Ceiling evidence from the commute anchor (the one time-use quantity ACS measures per tract): "
    "modeled vs measured tract commutes correlate at only r~0.21, so place-driven variation beyond "
    "composition is largely unseen — read tract values as compositional expectations, strongest "
    "for within-metro contrasts.")

ATUS_METRICS = {
    "modeled_leisure_minutes": {
        "combine": "weighted_mean", "value": "leisure_minutes", "weight": "adult_pop",
        "unit": "minutes_per_day", "direction": "higher_better", "synthetic": True,
        "label": "Modeled daily leisure time", "category": "time_balance",
        "caveat": ATUS_CAVEAT + " Note the composition effect: areas with more retirees or fewer "
                  "workers model MORE leisure — this is a time-budget descriptor, not a prosperity "
                  "score.",
    },
    "modeled_time_poverty_pct": {
        "combine": "weighted_mean", "value": "time_poverty_pct", "weight": "adult_pop",
        "unit": "percent", "direction": "lower_better", "synthetic": True,
        "label": "Modeled time poverty", "category": "time_balance",
        "caveat": ATUS_CAVEAT + " Time-poor = under 120 minutes (2h) of daily leisure — an absolute "
                  "threshold that lands within ~6% of the 50%-of-median relative line used in the "
                  "time-poverty literature.",
    },
}


def write_atus_spec(out_dir, geographies, *, vintage: str, atus_years: str, calibration=None,
                    validation=None):
    """Merge the ATUS time-use layers + metrics into aggregation_spec.json (sibling of the GSS
    outcome layers; layer `atus_<geo>` reads atus_<geo>.csv)."""
    out_dir = Path(out_dir)
    path = out_dir / "aggregation_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "contract": CONTRACT, "source": SOURCE, "synthetic_estimate": True}
    layers = spec.get("layers", {})
    for geo in geographies:
        layers[f"atus_{geo}"] = {"file": f"atus_{geo}.csv", "kind": "polygon_values",
                                 "id_field": "geoid"}
    all_atus = [n for n in layers if n.startswith("atus_")]
    spec["layers"] = layers
    metrics = spec.get("metrics", {})
    metrics.update({mid: {**m, "layer": all_atus, "ci": [f"{m['value']}_lo", f"{m['value']}_hi"]}
                    for mid, m in ATUS_METRICS.items()})
    spec["metrics"] = metrics
    spec["atus_outcomes"] = {
        "atus_years": atus_years, "acs_vintage": vintage, "confidence_interval": CI_NOTE,
        "model": ("circumstantial-only weighted MRP on pooled ATUS diary days (TUFNWGTP weights "
                  "correct the weekend oversampling; CPS-linked demographics); predictors = the "
                  "shared frame + a fulltime margin (ACS B23022); no PLACES margin (ATUS has no "
                  "linked health item). Leisure = weighted OLS (weighted zero-share ~5%, so no "
                  "hurdle); time poverty = weighted logit; the validation commute model is "
                  "two-part (67% zero days)."),
    }
    if calibration is not None:
        spec["atus_outcomes"]["calibration"] = {
            "method": "national additive benchmark to the TUFNWGTP-weighted pooled ATUS rate",
            "offsets": calibration}
    if validation is not None:
        spec["atus_outcomes"]["validation"] = validation
    path.write_text(json.dumps(spec, indent=2))
    return path


CI_NOTE = {
    "level": 0.90,
    "scope": "model_coefficient",
    "note": ("A 90% interval over the fitted GSS coefficients' sampling uncertainty only "
             "(<metric>_lo/_hi columns). It EXCLUDES the PLACES-margin and structural/synthetic "
             "uncertainty, which are larger — so treat it as a lower bound on total uncertainty."),
}


def write_spec(out_dir, geographies, *, vintage: str, model_key: str, gss_years: str, calibration=None):
    """Write/merge aggregation_spec.json next to the per-geography CSVs (happiness_<geo>.csv).

    Layers accumulate across calls (so running tract then zcta yields a spec with both). `calibration`
    records the national benchmarking offsets applied to the published values."""
    out_dir = Path(out_dir)
    path = out_dir / "aggregation_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    spec.update({
        "contract": CONTRACT, "source": SOURCE, "synthetic_estimate": True, "caveat": CAVEAT,
        "acs_vintage": vintage, "model": model_key, "gss_years": gss_years, "confidence_interval": CI_NOTE,
    })
    if calibration is not None:
        spec["calibration"] = {"method": "national additive benchmark to design-weighted GSS rate",
                               "offsets": calibration}
    layers = spec.get("layers", {})
    for geo in geographies:
        layers[geo] = {"file": f"happiness_{geo}.csv", "kind": "polygon_values", "id_field": "geoid"}
    spec["layers"] = layers
    metrics = spec.get("metrics", {})
    metrics.update({mid: {**m, "layer": list(layers), "ci": [f"{m['value']}_lo", f"{m['value']}_hi"]}
                    for mid, m in METRICS.items()})
    spec["metrics"] = metrics
    path.write_text(json.dumps(spec, indent=2))
    return path


def write_outcomes_spec(out_dir, geographies, *, vintage: str, gss_years: str, calibration=None,
                        calibration_window: str | None = None):
    """Merge the modeled-outcomes layers + metrics into aggregation_spec.json.

    Outcome layers are named `outcomes_<geo>` (file outcomes_<geo>.csv) so they coexist with the
    happiness layers in one spec; a consumer keys source geometries by these layer names. Any stale
    `religion_*` layers/metrics from the superseded religion-only artifact are dropped."""
    out_dir = Path(out_dir)
    path = out_dir / "aggregation_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "contract": CONTRACT, "source": SOURCE, "synthetic_estimate": True}
    spec.pop("religion", None)
    layers = {n: v for n, v in spec.get("layers", {}).items() if not n.startswith("religion_")}
    for geo in geographies:
        layers[f"outcomes_{geo}"] = {"file": f"outcomes_{geo}.csv", "kind": "polygon_values",
                                     "id_field": "geoid"}
    all_outcomes = [n for n in layers if n.startswith("outcomes_")]
    spec["layers"] = layers
    metrics = {mid: m for mid, m in spec.get("metrics", {}).items() if mid not in OUTCOME_METRICS
               and not str(m.get("layer", [""])[0]).startswith("religion_")}
    metrics.update({mid: {**m, "layer": all_outcomes, "ci": [f"{m['value']}_lo", f"{m['value']}_hi"]}
                    for mid, m in OUTCOME_METRICS.items()})
    spec["metrics"] = metrics
    spec["outcomes"] = {
        "gss_years": gss_years, "acs_vintage": vintage, "confidence_interval": CI_NOTE,
        "models": {
            "religion family (attendance, no_religion)":
                "identity-aware: circumstantial predictors + age4/sex/race_ethnicity, raked on ACS "
                "B01001 adult age x sex + B03002 race margins (race margin is all-ages — an "
                "approximation) — descriptive metrics, so identity is deliberately included",
            "fear_walking":
                "identity fitted but STANDARDIZED to the national mix (adjusts for response "
                "composition without mapping it) + log-density term fit on GSS SRCBELT anchors and "
                "projected on each area's actual density (density.py)",
            "wellbeing family (financial_satisfaction, social_trust)":
                "circumstantial-only, identity excluded by the same policy as the happiness metrics",
            "socializing family (weekly_friends/neighbors/bar)":
                "circumstantial + log-density; orders place types, not regions — see caveats",
        },
    }
    if calibration is not None:
        spec["outcomes"]["calibration"] = {
            "method": "national additive benchmark to design-weighted GSS rate, RECENT window "
                      "(attendance, trust, and socializing decline secularly, so pooled rates would "
                      "overstate today's levels)",
            "window": calibration_window, "offsets": calibration}
    path.write_text(json.dumps(spec, indent=2))
    return path
