"""County Business Patterns (ZBP): ZIP-level establishment counts — the LOI/supply-side covariates.

CBP publishes ANNUAL establishment counts by NAICS at ZIP level (1998-present; ZIP≈ZCTA is the
standard approximation). First use: religious congregations (NAICS 8131) per 1,000 adults as an
area covariate in the attendance model — the supply side of the behavior being modeled, and the
route by which denominational geography (e.g. Utah) can enter, which pure composition cannot see.

API vintages: 2017+ live under /data/{year}/cbp with NAICS2017; 2012-2016 under /data/{year}/zbp
with NAICS2012. Responses are cached raw (ABC). Needs CENSUS_API_KEY (acs.census_key, .env-aware).

Tract values: ZIP counts land on ZCTAs; each tract inherits its dominant ZCTA's per-capita rate via
the Census tract<->ZCTA relationship file (largest land-overlap ZCTA wins) — a v1 approximation.
"""

from __future__ import annotations

import json
from urllib.request import urlopen

import numpy as np
import pandas as pd

from microhappiness.acs import CACHE_DIR, census_key

NAICS_RELIGIOUS = "8131"
_REL_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_tract20_natl.txt"
# log10(rate + _RATE_FLOOR): keeps zero-congregation areas finite without dominating the scale.
_RATE_FLOOR = 0.1


def fetch_zip_establishments(naics: str = NAICS_RELIGIOUS, year: int = 2023) -> dict:
    """{zip: establishment count} for one NAICS code, all ZIPs, from the CBP/ZBP API (cached)."""
    cache = CACHE_DIR / f"cbp_{naics}_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if year >= 2017:
        url = (f"https://api.census.gov/data/{year}/cbp?get=ESTAB&for=zip%20code:*"
               f"&NAICS2017={naics}&key={census_key()}")
    else:
        url = (f"https://api.census.gov/data/{year}/zbp?get=ESTAB&for=zipcode:*"
               f"&NAICS2012={naics}&key={census_key()}")
    rows = json.loads(urlopen(url, timeout=600).read())
    header = rows[0]
    zi, ei = (len(header) - 1), header.index("ESTAB")
    out = {r[zi]: int(r[ei]) for r in rows[1:] if r[ei] not in (None, "")}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def tract_to_zcta() -> dict:
    """{tract geoid: dominant ZCTA} from the 2020 Census relationship file (largest land overlap)."""
    cache = CACHE_DIR / "tract_zcta_rel2020.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    df = pd.read_csv(_REL_URL, sep="|", dtype=str,
                     usecols=["GEOID_ZCTA5_20", "GEOID_TRACT_20", "AREALAND_PART"])
    df["AREALAND_PART"] = pd.to_numeric(df["AREALAND_PART"], errors="coerce").fillna(0)
    df = df.dropna(subset=["GEOID_ZCTA5_20"])
    best = df.sort_values("AREALAND_PART").groupby("GEOID_TRACT_20").tail(1)
    out = dict(zip(best["GEOID_TRACT_20"], best["GEOID_ZCTA5_20"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def log_congregation_rate(geography: str = "tract", year: int = 2023, acs_year: int = 2022) -> dict:
    """{geoid: log10(congregations per 1,000 adults + floor)} for tracts or ZCTAs."""
    from microhappiness.density import _population  # adult-ish totals; B01001_001E all-ages

    counts = fetch_zip_establishments(NAICS_RELIGIOUS, year)
    pop_z = _population("zcta", acs_year)
    rate_z = {}
    for z, p in pop_z.items():
        if p and p > 0:
            rate_z[z] = np.log10(counts.get(z, 0) / p * 1000.0 + _RATE_FLOOR)
    if geography == "zcta":
        return rate_z
    t2z = tract_to_zcta()
    return {t: rate_z[z] for t, z in t2z.items() if z in rate_z}


def cell_rates(gss_binned, rate_zcta: dict, log_dens_zcta: dict, pop_zcta: dict, anchors: dict) -> dict:
    """Fit-side attachment: {(region, srcbelt): mean log-congregation-rate} over ZCTAs.

    GSS respondents carry only (region, belt); each ZCTA gets a pseudo-belt (nearest density anchor)
    and a region (via its dominant tract's state, from the relationship file — ZCTAs don't nest in
    states), and the pop-weighted cell mean becomes the respondent's covariate value — the same
    ecological bridge as density.py, one level up."""
    from microhappiness.validate import _REGION_BY_STATE

    region_by_z = {z: _REGION_BY_STATE.get(t[:2]) for t, z in tract_to_zcta().items()}
    belt_of = lambda ld: min(anchors, key=lambda b: abs(anchors[b] - ld))
    acc: dict = {}
    for z, r in rate_zcta.items():
        ld, region = log_dens_zcta.get(z), region_by_z.get(z)
        if ld is None or region is None:
            continue
        key = (region, float(belt_of(ld)))
        w = pop_zcta.get(z, 0.0)
        s, n = acc.get(key, (0.0, 0.0))
        acc[key] = (s + w * r, n + w)
    return {k: s / n for k, (s, n) in acc.items() if n > 0}
