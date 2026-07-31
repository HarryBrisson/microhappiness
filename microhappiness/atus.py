"""ATUS microdata: pool ~10 years of American Time Use Survey diary days onto the repo's frame.

Structural differences from the GSS adapter (gss.py) that shape everything here:
- Each respondent contributes ONE diary day. The ATUS final weight TUFNWGTP is a person-DAY weight
  built to handle the deliberate weekend oversampling (~50% of diaries are weekend days vs 2/7 of
  real days), so every weighted statistic — fits, the poststrat seed, calibration targets — must use
  it. The multi-year file's weights are BLS's pooled-period weights (they also absorb the 2020
  collection gap); we use them as published.
- Demographics come from the linked CPS variables (atuscps file, TULINENO==1 = the ATUS respondent),
  recoded onto the same binned circumstantial frame as gss.py/binning.py so the ACS raking margins
  are shared. Income is the CPS 16-bracket family income mapped straight onto the ACS B19001 dollar
  brackets (<25k/25-50k/50-100k/100k+) — cleaner than GSS's percentile bridge, but pooling 2016-2025
  mixes nominal dollars across years (a documented approximation; national levels are calibrated).
- Public-use geography is richer than GSS: state FIPS + metro status per respondent, which the
  step-0 gate uses for holdout ordering (GSS only ever revealed region + size-class belt).

Time-use constructs (minutes per diary day, from the activity summary file):
- leisure_min: BLS's "Leisure and sports" measure = major categories 12 (socializing, relaxing, and
  leisure) + 13 (sports, exercise, and recreation). We follow the BLS convention and INCLUDE
  category 13 — published averages are then directly comparable to the ATUS news-release tables;
  category-12-only is kept as social_leisure_min for sensitivity checks.
- commute_min: travel related to work (t1805xx). Diary-day commuting, zero on non-work days — the
  validation anchor against ACS-measured tract commutes, never a published metric.

Source: https://www.bls.gov/tus/data.htm multi-year files (atussum/atusresp/atuscps), cached as raw
zips in data/atus_cache/ (ABC rule: the ~130MB of raw downloads persist; parsing is re-runnable).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/atus_cache")
ATUS_URL = "https://www.bls.gov/tus/datafiles/{name}.zip"
_UA = "microhappiness/0.0 (research; https://github.com/HarryBrisson/microhappiness)"

# Multi-year 2003-2025 files (2025 released 2026-06). POOL_YEARS trims to the modeled decade.
# 2020 diary days carry TUFNWGTP = -1 in the multi-year files (BLS: the pandemic collection gap
# makes 2020 incomparable under the pooled weight), so load_atus drops them and the pool spans
# 2015-2025 to keep ten usable diary years.
FILES = ("atussum-0325", "atusresp-0325", "atuscps-0325")
POOL_YEARS = (2015, 2025)

# The circumstantial predictor frame shared with the ACS margins. Same bins as binning.py plus
# `fulltime` (works 35+ hours), which GSS never carried cleanly but is THE load-bearing time-use
# predictor; its ACS margin comes from B23022 (acs.fetch_fulltime_share_sf).
ATUS_PREDICTORS = ("married", "employment", "home_owner", "lives_alone", "income4", "fulltime")

# CPS HEFAMINC 16 family-income brackets -> the 4 ACS B19001 groups (<25k/25-50k/50-100k/100k+).
_HEFAMINC_TO_INCOME4 = {**{c: 0.0 for c in range(1, 8)}, **{c: 1.0 for c in range(8, 12)},
                        **{c: 2.0 for c in range(12, 15)}, **{c: 3.0 for c in (15, 16)}}


def download_atus(name: str, dest_dir: str | Path = CACHE_DIR) -> Path:
    """Fetch one BLS multi-year zip to the cache. Idempotent (the raw zip IS the cache)."""
    dest = Path(dest_dir) / f"{name}.zip"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = urlopen(Request(ATUS_URL.format(name=name), headers={"User-Agent": _UA}),
                   timeout=1800).read()
    dest.write_bytes(blob)
    return dest


def _read_zip_csv(zip_path: Path, **kwargs) -> pd.DataFrame:
    """Read the single .dat (comma-separated) inside a BLS zip."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".dat"))
        with zf.open(name) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), **kwargs)


def _activity_columns(zip_path: Path) -> dict[str, list[str]]:
    """Split the summary file's tXXYYZZ columns into our constructs (robust to code additions)."""
    header = _read_zip_csv(zip_path, nrows=0).columns
    return {
        "leisure_min": [c for c in header if c.startswith(("t12", "t13"))],
        "social_leisure_min": [c for c in header if c.startswith("t12")],
        "commute_min": [c for c in header if c.startswith("t1805")],
    }


def load_atus(cache_dir: str | Path = CACHE_DIR, years: tuple[int, int] = POOL_YEARS) -> pd.DataFrame:
    """One row per respondent diary day: weight, time-use minutes, binned predictors, geography.

    Adults (18+) only, to match the ACS margins' universes. Rows keep NaN predictors (fits
    complete-case per model, as in outcomes.py).
    """
    cache_dir = Path(cache_dir)
    sum_zip, resp_zip, cps_zip = (download_atus(n, cache_dir) for n in FILES)

    acts = _activity_columns(sum_zip)
    act_cols = sorted({c for cols in acts.values() for c in cols})
    d = _read_zip_csv(sum_zip, usecols=["TUCASEID", "TUYEAR", "TUFNWGTP", *act_cols])
    d = d[d["TUYEAR"].between(*years) & (d["TUFNWGTP"] > 0)].copy()  # w=-1: 2020, unusable pooled
    for construct, cols in acts.items():
        d[construct] = d[cols].sum(axis=1)
    d = d.drop(columns=act_cols)

    resp = _read_zip_csv(resp_zip, usecols=["TUCASEID", "TUDIARYDAY", "TELFS", "TRDPFTPT"])
    cps = _read_zip_csv(cps_zip, usecols=["TUCASEID", "TULINENO", "PRTAGE", "PEMARITL", "HETENURE",
                                          "HRNUMHOU", "HEFAMINC", "GEREG", "GESTFIPS", "GTMETSTA"])
    cps = cps[cps["TULINENO"] == 1].drop(columns="TULINENO")  # the ATUS respondent's own CPS record
    d = d.merge(resp, on="TUCASEID").merge(cps, on="TUCASEID")
    d.columns = [c.lower() for c in d.columns]
    d = d[d["prtage"] >= 18].rename(columns={"tuyear": "year"})

    # Recodes onto the shared frame (see gss.recode_predictors for the GSS siblings).
    d["married"] = d["pemaritl"].map({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0})
    d["employment"] = d["telfs"].map({1: "employed", 2: "employed", 3: "unemployed",
                                      4: "unemployed", 5: "nilf"})
    d["fulltime"] = np.where(d["trdpftpt"] == 1, 1.0,
                             np.where(d["telfs"].isin([1, 2, 3, 4, 5]), 0.0, np.nan))
    d["home_owner"] = d["hetenure"].map({1: 1.0, 2: 0.0, 3: 0.0})
    hh = pd.to_numeric(d["hrnumhou"], errors="coerce")
    d["lives_alone"] = (hh == 1).where(hh >= 1).astype("float")
    d["income4"] = d["hefaminc"].map(_HEFAMINC_TO_INCOME4)
    # Holdout-validation geography (public-use: state + census region + metro status).
    d["region"] = d["gereg"]
    d["state_fips"] = d["gestfips"]
    d["metro"] = d["gtmetsta"].where(d["gtmetsta"].isin([1, 2]))  # 1=metro 2=nonmetro (3=unidentified)
    return d.drop(columns=["pemaritl", "telfs", "trdpftpt", "hetenure", "hrnumhou", "hefaminc",
                           "gereg", "gestfips", "gtmetsta"])


def weighted_stats(d: pd.DataFrame, col: str, w: str = "tufnwgtp") -> dict:
    """Weighted mean/median/zero-share for a minutes construct (diagnostics + threshold documentation)."""
    x = d[col].to_numpy(float)
    wt = d[w].to_numpy(float)
    order = np.argsort(x)
    cum = np.cumsum(wt[order])
    median = float(x[order][np.searchsorted(cum, cum[-1] / 2)])
    return {"mean": float(np.average(x, weights=wt)), "median": median,
            "zero_share": float(np.average(x == 0, weights=wt)), "n": int(len(x))}
