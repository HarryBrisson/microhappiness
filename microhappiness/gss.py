"""GSS microdata: load the cumulative datafile and recode predictors to match ACS categories.

The recodes are the contract between survey and census: a GSS `marital` category must mean the same
thing as the ACS B12001 bucket it'll be poststratified against. Keep these aligned with acs.py.

Source: GSS cumulative datafile 1972-present (https://gss.norc.org/get-the-data), Stata/SPSS. Or the
`gssr` R package. Public use; geography is region-only (which is *why* we model + poststratify).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

# GSS columns we pull (cumulative-file names, lowercased on load).
GSS_COLUMNS = (
    "year", "happy", "marital", "age", "sex", "race", "hispanic",
    "wrkstat", "educ", "degree", "realinc", "income", "region",
    "health", "mntlhlth",           # self-rated health + poor-mental-health days (some years)
    "dwelown", "hompop", "childs", "hrs1",  # GNH domains: home ownership / household / kids / time
    "born", "vetyears", "physhlth", "smoke", "wrkslf", "prestg10",  # more ACS/PLACES-analog constructs
    "wtssps", "wtssall",
)

HAPPY_RECODE = {1: 3, 2: 2, 3: 1}  # GSS: 1=very,2=pretty,3=not too  ->  3/2/1 (higher = happier)

# GSS cumulative cross-sectional datafile (1972-present). The exact asset URL changes per release;
# override with MICROHAPPINESS_GSS_URL or just hand `download_gss` the current link / drop the file.
GSS_CUMULATIVE_URL = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip"


def download_gss(dest: str | Path = "data/gss_cumulative.dta", url: str | None = None) -> Path:
    """Fetch the GSS cumulative Stata file (zip or raw .dta) to `dest`. Idempotent."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url or GSS_CUMULATIVE_URL
    blob = urlopen(Request(url, headers={"User-Agent": "microhappiness/0.0"}), timeout=600).read()
    if url.endswith(".zip") or blob[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".dta"))
            dest.write_bytes(zf.read(name))
    else:
        dest.write_bytes(blob)
    return dest


def load_gss(path: str | Path, columns: tuple[str, ...] | None = GSS_COLUMNS) -> pd.DataFrame:
    """Read the GSS cumulative .dta/.sav, keep only the columns we need, as RAW numeric codes.

    `convert_categoricals=False` is essential: it returns integer codes (which the recodes below
    expect) and skips value-label conversion — the GSS cumulative file has columns whose labels aren't
    pandas-categorical-safe, and reading only our columns sidesteps them entirely.
    """
    path = Path(path)
    want = [c.lower() for c in columns] if columns else None
    if path.suffix.lower() != ".dta":
        df = pd.read_spss(path)
        df.columns = [c.lower() for c in df.columns]
        return df[[c for c in (want or df.columns) if c in df.columns]].copy()
    # Discover available columns cheaply (header + 1 row), then read just the ones we want.
    with pd.read_stata(path, convert_categoricals=False, chunksize=1) as reader:
        available = [c.lower() for c in next(reader).columns]
    use = [c for c in (want or available) if c in available]
    df = pd.read_stata(path, columns=use, convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def recode_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw GSS codes onto the ACS-aligned categories named in models.py.

    NOTE: bucket boundaries below are the design intent; verify exact GSS code values against the
    current cumulative codebook at build time (they are stable but worth confirming once).
    """
    out = df.copy()
    out["happy"] = out["happy"].map(HAPPY_RECODE)

    # marital -> {married, prev_married, never_married} to match ACS B12001 collapsing.
    out["marital"] = out["marital"].map({
        1: "married", 2: "prev_married", 3: "prev_married", 4: "prev_married", 5: "never_married",
    })
    # employment (WRKSTAT) -> {full_time, other} to match ACS B23025 (employed full-time vs rest).
    out["employment"] = out["wrkstat"].map(lambda v: "full_time" if v == 1 else "other")
    # education -> years (EDUC is already 0-20); also a degree bucket for ACS B15003 crosswalk.
    out["education"] = out["educ"]
    # race/ethnicity -> {hispanic, white_nh, black_nh, other_nh} to match ACS B03002.
    out["race_ethnicity"] = _race_ethnicity(out)
    # income -> decile of REALINC within year (cross-sectional rank is the load-bearing signal).
    out["income"] = out.groupby("year")["realinc"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["sex"] = out["sex"].map({1: "male", 2: "female"})
    # self-rated health -> fair-or-poor binary, to align with PLACES GHLTH (1=excel..4=poor).
    if "health" in out:
        out["health"] = out["health"].map({1: 0, 2: 0, 3: 1, 4: 1})  # 1 = fair-or-poor
    # mental health -> >=14 poor-mental-health days, to align with PLACES MHLTH (asked some years only).
    # Preserve NaN where MNTLHLTH wasn't asked — otherwise those respondents look falsely "good".
    if "mntlhlth" in out:
        mh = pd.to_numeric(out["mntlhlth"], errors="coerce")
        out["mental_health"] = (mh >= 14).where(mh.notna()).astype("Int64")
    # GNH-domain candidates, each fittable in GSS AND reproducible locally (ACS):
    if "dwelown" in out:                                            # material wellbeing  <-> ACS B25003
        out["home_owner"] = out["dwelown"].map({1: 1, 2: 0, 3: 0}).astype("Int64")
    if "hompop" in out:                                            # social connectedness <-> ACS B11001
        hp = pd.to_numeric(out["hompop"], errors="coerce")
        out["lives_alone"] = (hp == 1).where(hp >= 1).astype("Int64")
    if "childs" in out:                                            # material/social      <-> ACS B09002
        ch = pd.to_numeric(out["childs"], errors="coerce")
        out["num_children"] = ch.where(ch <= 8)                    # GSS tops at "8 or more"; 9=NA
    if "hrs1" in out:                                              # time balance         <-> ACS B23022
        hr = pd.to_numeric(out["hrs1"], errors="coerce")
        out["hours_worked"] = hr.where((hr >= 0) & (hr <= 89))     # NaN for non-workers / DK
    # Further ACS/PLACES-analog constructs (for the full transparency catalog):
    if "born" in out:                                              # nativity             <-> ACS B05002
        out["us_born"] = out["born"].map({1: 1, 2: 0}).astype("Int64")
    if "vetyears" in out:                                          # veteran status       <-> ACS B21001
        vy = pd.to_numeric(out["vetyears"], errors="coerce")
        out["veteran"] = (vy > 1).where(vy.isin([1, 2, 3, 4])).astype("Int64")  # 1=none,2-4=served
    if "physhlth" in out:                                          # physical health      <-> PLACES PHLTH
        ph = pd.to_numeric(out["physhlth"], errors="coerce")
        out["poor_phys_health"] = (ph >= 14).where(ph.notna()).astype("Int64")
    if "smoke" in out:                                             # smoking              <-> PLACES CSMOKING
        out["smoker"] = out["smoke"].map({1: 1, 2: 0}).astype("Int64")
    if "wrkslf" in out:                                            # self-employment      <-> ACS B24080
        out["self_employed"] = out["wrkslf"].map({1: 1, 2: 0}).astype("Int64")
    if "prestg10" in out:                                          # occ. prestige        <-> ACS occupation
        pr = pd.to_numeric(out["prestg10"], errors="coerce")
        out["occ_prestige"] = pr.where(pr > 0)
    return out


def _race_ethnicity(df: pd.DataFrame) -> pd.Series:
    hisp = df.get("hispanic", pd.Series(1, index=df.index)) != 1  # HISPANIC==1 is "not hispanic"
    race = df["race"]  # 1=white, 2=black, 3=other
    out = pd.Series("other_nh", index=df.index)
    out[race == 1] = "white_nh"
    out[race == 2] = "black_nh"
    out[hisp] = "hispanic"
    return out
