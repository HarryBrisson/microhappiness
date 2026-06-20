"""ACS via the Census API: per-tract marginal proportions for the v1 poststratification bins.

Comprehensive on the data side is the v2 goal; v1 pulls exactly the tables the binned predictors need
and returns, per tract GEOID, a proportion vector for each margin. Categories must mirror binning.py.

Census API: https://api.census.gov/data/<year>/acs/acs5  (key required, CENSUS_API_KEY).
"""

from __future__ import annotations

import json
import os
from urllib.request import urlopen

import pandas as pd

API = "https://api.census.gov/data/{year}/acs/acs5"

# Table -> the variables we sum into each bin (codes verified against the ACS group metadata).
MARRIED = ["B12001_004E", "B12001_013E"]          # now-married male + female / B12001_001E
TENURE = {"owner": "B25003_002E", "total": "B25003_001E"}
HH = {"alone": "B11001_008E", "total": "B11001_001E"}
EMP = {"employed": ["B23025_004E", "B23025_006E"], "unemployed": ["B23025_005E"],
       "nilf": ["B23025_007E"]}                    # armed forces folded into employed
INCOME_BINS = {                                     # B19001 brackets -> 4 income groups
    0: [f"B19001_{i:03d}E" for i in range(2, 6)],   # <25k  (<10,10-15,15-20,20-25)
    1: [f"B19001_{i:03d}E" for i in range(6, 11)],  # 25-50k
    2: [f"B19001_{i:03d}E" for i in range(11, 14)], # 50-100k
    3: [f"B19001_{i:03d}E" for i in range(14, 18)], # 100k+
}
INCOME_TOTAL = "B19001_001E"

_VARS = (["B12001_001E", *MARRIED, TENURE["owner"], TENURE["total"], HH["alone"], HH["total"],
          *sum(EMP.values(), []), INCOME_TOTAL]
         + [v for vs in INCOME_BINS.values() for v in vs])


def census_key() -> str:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("Set CENSUS_API_KEY (https://api.census.gov/data/key_signup.html)")
    return key


def _get(year, variables, key, *, geography="tract", state=None):
    if geography == "tract":
        geo = f"&for=tract:*&in=state:{state}"
    else:  # zcta — national, not nested in state
        geo = "&for=zip%20code%20tabulation%20area:*"
    url = API.format(year=year) + "?get=" + ",".join(variables) + geo + f"&key={key}"
    rows = json.loads(urlopen(url, timeout=600).read())
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for v in variables:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df["geoid"] = (df["state"] + df["county"] + df["tract"] if geography == "tract"
                   else df["zip code tabulation area"])
    return df.set_index("geoid")


def fetch_acs_margins(state: str | None = None, year: int = 2022, key: str | None = None,
                      geography: str = "tract") -> dict:
    """{geoid: {margin_name: proportion-vector}} for married/employment/home_owner/lives_alone/income4.

    geography="tract" needs a state FIPS; geography="zcta" is national. Each margin is a dict of
    bin->proportion summing to ~1 (rows with a zero denominator are dropped).
    """
    key = key or census_key()
    df = _get(year, _VARS, key, geography=geography, state=state)
    out = {}
    for geoid, r in df.iterrows():
        m = {}
        tot = r["B12001_001E"]
        if tot and tot > 0:
            p = (r[MARRIED[0]] + r[MARRIED[1]]) / tot
            m["married"] = {1.0: p, 0.0: 1 - p}
        et = sum(r[v] for vs in EMP.values() for v in vs)
        if et and et > 0:
            m["employment"] = {k: sum(r[v] for v in vs) / et for k, vs in EMP.items()}
        if r[TENURE["total"]] and r[TENURE["total"]] > 0:
            p = r[TENURE["owner"]] / r[TENURE["total"]]
            m["home_owner"] = {1.0: p, 0.0: 1 - p}
        if r[HH["total"]] and r[HH["total"]] > 0:
            p = r[HH["alone"]] / r[HH["total"]]
            m["lives_alone"] = {1.0: p, 0.0: 1 - p}
        it = r[INCOME_TOTAL]
        if it and it > 0:
            m["income4"] = {float(b): sum(r[v] for v in vs) / it for b, vs in INCOME_BINS.items()}
        if len(m) == 5:  # keep only fully-populated tracts
            out[geoid] = m
    return out
