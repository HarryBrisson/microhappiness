"""CDC PLACES tract health estimates — the poststratification margin ACS can't give us.

PLACES publishes modeled health prevalence at tract AND ZCTA (our exact geography). Its value here is
specific: self-rated GENERAL health (% fair-or-poor, measure GHLTH) is one of the strongest happiness
predictors ACS lacks. We fit happiness ~ health on GSS HEALTH and rake the PLACES margin into each
area's joint table (the GSS seed carries the circumstance x health correlation).

So: ACS = circumstantial margins, PLACES = the health margin, GSS = the model AND the joint seed.

CAUTIONS (see METHODOLOGY.md): synthetic-on-synthetic (PLACES is itself a BRFSS->ACS MRP, so uncertainty
compounds — label it); align GSS HEALTH to PLACES 'fair-or-poor'. Mental health (MHLTH) is a DISTINCT
axis (not circular) usable as a predictor (M5) where a GSS analog exists; depression -> validation only.

Source: PLACES tract release on data.cdc.gov (SODA resource cwsq-ngmh): locationname=tract GEOID,
data_value=prevalence %, totalpopulation=adult population.
"""

from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import urlopen

RESOURCE = {"tract": "https://data.cdc.gov/resource/cwsq-ngmh.json",   # 2025 release: 2020 tracts (matches ACS)
            "zcta": "https://data.cdc.gov/resource/qnzd-25i4.json"}    # 2025 ZCTA release
# The 2025 tract release OMITS Kentucky + Pennsylvania (they fall out of the recent BRFSS cycles), which
# blanked those two states on the national map. Backfill them from the 2023 release — a documented
# per-state vintage compromise (2023 uses 2010 tracts, so a fraction won't join the 2020 ACS) that beats
# leaving two whole states empty.
_TRACT_FALLBACK = "https://data.cdc.gov/resource/em5e-5hvn.json"
_FALLBACK_STATES = ("KY", "PA")
PREDICTOR_MEASURES = {"GHLTH": "fair-or-poor self-rated health", "MHLTH": "frequent mental distress",
                      "CSMOKING": "current smoking"}
VALIDATION_MEASURES = {"MHLTH": "mental distress (validates M1-M4)", "DEPRESSION": "depression (validation only)"}


def _pull(resource: str, measure: str, state_abbr: str | None) -> dict:
    out: dict = {}
    offset, page = 0, 50000
    where = f"stateabbr={state_abbr}&" if state_abbr else ""
    while True:
        # $order=:id keeps Socrata's offset paging stable (without a sort it can overlap/drop rows).
        url = (f"{resource}?{where}measureid={measure}&$order=:id"
               f"&$select=locationname,data_value,totalpopulation&$limit={page}&$offset={offset}")
        rows = json.loads(urlopen(url, timeout=600).read())
        for r in rows:
            v, geoid = r.get("data_value"), r.get("locationname")
            if v is None or not geoid:
                continue
            out[geoid] = {"fraction": float(v) / 100.0, "adult_pop": float(r.get("totalpopulation") or 0)}
        if len(rows) < page:
            return out
        offset += page


def fetch_measure(measure: str = "GHLTH", *, geography: str = "tract", state_abbr: str | None = None) -> dict:
    """{geoid: {'fraction': p, 'adult_pop': n}} for a PLACES measure. National unless state_abbr given.

    For a national tract pull, KY/PA (absent from the 2025 release) are backfilled from the 2023 release.
    """
    out = _pull(RESOURCE[geography], measure, state_abbr)
    if geography == "tract" and state_abbr is None:
        for st in _FALLBACK_STATES:
            for geoid, rec in _pull(_TRACT_FALLBACK, measure, st).items():
                out.setdefault(geoid, rec)  # fill only what the primary release lacked
    return out
