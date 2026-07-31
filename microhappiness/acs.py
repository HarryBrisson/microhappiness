"""ACS via the Census API: per-tract marginal proportions for the v1 poststratification bins.

Comprehensive on the data side is the v2 goal; v1 pulls exactly the tables the binned predictors need
and returns, per tract GEOID, a proportion vector for each margin. Categories must mirror binning.py.

Census API: https://api.census.gov/data/<year>/acs/acs5  (key required, CENSUS_API_KEY).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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

# Identity margins (outcomes.py only — the happiness models never rake these). B01001 adult age×sex
# cells: male 007..025, female 031..049; bins mirror binning.AGE4_CUTS (18-34/35-49/50-64/65+).
_A = lambda lo, hi: [f"B01001_{i:03d}E" for i in range(lo, hi + 1)]
AGE4_BINS = {0.0: _A(7, 12) + _A(31, 36), 1.0: _A(13, 15) + _A(37, 39),
             2.0: _A(16, 19) + _A(40, 43), 3.0: _A(20, 25) + _A(44, 49)}
SEX_BINS = {"male": _A(7, 25), "female": _A(31, 49)}
# B03002 (all ages — an approximation for the adult joint, noted in METHODOLOGY): white/black NH,
# hispanic; other_nh is the remainder.
RACE = {"total": "B03002_001E", "white_nh": "B03002_003E", "black_nh": "B03002_004E",
        "hispanic": "B03002_012E"}
_IDENTITY_VARS = sorted({v for vs in AGE4_BINS.values() for v in vs}
                        | set(SEX_BINS["male"]) | set(SEX_BINS["female"]) | set(RACE.values()))


def census_key() -> str:
    key = os.environ.get("CENSUS_API_KEY") or _dotenv_key()
    if not key:
        raise RuntimeError("Set CENSUS_API_KEY (https://api.census.gov/data/key_signup.html) — "
                           "or use fetch_acs_margins_sf(), which needs no key")
    return key


def _dotenv_key() -> str:
    """CENSUS_API_KEY from a local .env (gitignored), so runs don't need the shell env set up."""
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("CENSUS_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


# Raw-response cache (ABC: persist every expensive call's raw output; re-runs read from disk).
CACHE_DIR = Path(os.environ.get("MICROHAPPINESS_ACS_CACHE", "data/acs_cache"))


def _get(year, variables, key, *, geography="tract", state=None):
    if geography == "tract":
        geo = f"&for=tract:*&in=state:{state}"
    else:  # zcta — national, not nested in state
        geo = "&for=zip%20code%20tabulation%20area:*"
    vhash = hashlib.sha1(",".join(variables).encode()).hexdigest()[:8]  # invalidate if the pull changes
    cache = CACHE_DIR / f"acs{year}_{geography}_{state or 'us'}_{vhash}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        url = API.format(year=year) + "?get=" + ",".join(variables) + geo + f"&key={key}"
        rows = json.loads(urlopen(url, timeout=600).read())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")
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
    return _margins_from_frame(df)


def _margins_from_frame(df) -> dict:
    """{geoid: margins} from a frame of the _VARS columns indexed by geoid (shared by API + SF paths)."""
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


# ---- Keyless path: the table-based ACS summary files (plain HTTPS, no API key) --------------------
# One national .dat per table covering every summary level; we filter to tract/ZCTA rows. The API
# now hard-requires a key, so this is the no-credential route; files are cached in data/acs_cache/.
SF_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/{year}"
          "/table-based-SF/data/5YRData/acsdt5y{year}-{table}.dat")
_SF_GEO_PREFIX = {"tract": "1400000US", "zcta": "860Z200US"}


def _sf_column(api_var: str) -> str:
    """API variable name -> summary-file column: B12001_004E -> B12001_E004."""
    table, rest = api_var.split("_")
    return f"{table}_E{rest[:-1]}"


def _sf_table_path(table: str, year: int) -> Path:
    import shutil

    dest = CACHE_DIR / f"acsdt5y{year}-{table.lower()}.dat"
    if not dest.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with urlopen(SF_URL.format(year=year, table=table.lower()), timeout=1800) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    return dest


def fetch_acs_margins_sf(year: int = 2022, geography: str = "tract",
                         include_identity: bool = False) -> dict:
    """The same margins as fetch_acs_margins, but NATIONAL in one pass from the keyless summary files.

    include_identity=True adds age4/sex/race_ethnicity margins (B01001 adult cells + B03002) for the
    identity-aware outcomes; the happiness pipeline never requests them."""
    variables = _VARS + (_IDENTITY_VARS if include_identity else [])
    tables = sorted({v.split("_")[0] for v in variables})
    prefix = _SF_GEO_PREFIX[geography]
    merged = None
    for table in tables:
        cols = ["GEO_ID"] + [_sf_column(v) for v in variables if v.startswith(table + "_")]
        df = pd.read_csv(_sf_table_path(table, year), sep="|", usecols=cols, dtype={"GEO_ID": str})
        df = df[df["GEO_ID"].str.startswith(prefix)]
        merged = df if merged is None else merged.merge(df, on="GEO_ID", how="inner")
    merged["geoid"] = merged["GEO_ID"].str[len(prefix):]
    merged = merged.set_index("geoid").rename(
        columns={_sf_column(v): v for v in variables}).apply(pd.to_numeric, errors="coerce")
    out = _margins_from_frame(merged)
    if include_identity:
        for geoid, r in merged.iterrows():
            m = out.get(geoid)
            if m is None:
                continue
            adults = sum(r[v] for vs in AGE4_BINS.values() for v in vs)
            if adults and adults > 0:
                m["age4"] = {b: sum(r[v] for v in vs) / adults for b, vs in AGE4_BINS.items()}
                m["sex"] = {s: sum(r[v] for v in vs) / adults for s, vs in SEX_BINS.items()}
            tot = r[RACE["total"]]
            if tot and tot > 0:
                shares = {k: r[v] / tot for k, v in RACE.items() if k != "total"}
                shares["other_nh"] = max(0.0, 1.0 - sum(shares.values()))
                m["race_ethnicity"] = shares
    return out
