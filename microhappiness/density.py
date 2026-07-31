"""Tract/ZCTA population density — the within-city urbanicity signal (Harry's ask, 2026-07).

Public GSS never reveals a respondent's density; it reveals their PLACE-TYPE (SRCBELT: central city /
suburb of top-12 vs mid-size metros / other urban / rural). We bridge ecologically:

  fit:      respondents carry the TYPICAL tract log10-density of their belt (belt anchors below), so
            the model learns d(outcome)/d(log density) from the 6 belt aggregates — validated: a
            linear log-density term recovers ~90-99% of the categorical C(srcbelt) fit.
  predict:  each tract carries its ACTUAL log10-density (Census Gazetteer land area x ACS population),
            so the projection varies tract-to-tract WITHIN a city — which the belt itself cannot.

Belt anchors are share-matched: order belts rural -> big-city core, split the national pop-weighted
tract density distribution into cumulative bands matching the GSS belt population shares, and anchor
each belt at its band's pop-weighted median. Tract densities are winsorized to the 1st-99th
pop-weighted percentiles so ultra-dense tracts don't extrapolate far beyond the fitted range.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from microhappiness.acs import CACHE_DIR, _sf_table_path, _SF_GEO_PREFIX

GAZETTEER_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/{year}_Gazetteer/"
                 "{year}_Gaz_{kind}_national.zip")
_GAZ_KIND = {"tract": "tracts", "zcta": "zcta"}
# Belts ordered least -> most urban for the share-matched anchoring (SRCBELT codes:
# 6 rural, 5 other urban, 4 suburb mid-metro, 3 suburb top-12, 2 central city mid, 1 central city top-12).
BELT_URBAN_ORDER = (6, 5, 4, 3, 2, 1)


def _gazetteer(geography: str, year: int = 2022) -> pd.DataFrame:
    dest = CACHE_DIR / f"gaz_{geography}_{year}.txt"
    if not dest.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url = GAZETTEER_URL.format(year=year, kind=_GAZ_KIND[geography])
        blob = urlopen(Request(url, headers={"User-Agent": "microhappiness/0.0"}), timeout=600).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            dest.write_bytes(zf.read(zf.namelist()[0]))
    df = pd.read_csv(dest, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    id_col = "GEOID" if "GEOID" in df.columns else "GEOID20"
    return pd.DataFrame({"geoid": df[id_col].str.strip(),
                         "aland_km2": pd.to_numeric(df["ALAND"], errors="coerce") / 1e6})


def _population(geography: str, year: int = 2022) -> pd.Series:
    """{geoid: total population} from the cached B01001 summary file."""
    prefix = _SF_GEO_PREFIX[geography]
    df = pd.read_csv(_sf_table_path("B01001", year), sep="|", usecols=["GEO_ID", "B01001_E001"],
                     dtype={"GEO_ID": str})
    df = df[df["GEO_ID"].str.startswith(prefix)]
    return pd.Series(pd.to_numeric(df["B01001_E001"], errors="coerce").to_numpy(),
                     index=df["GEO_ID"].str[len(prefix):])


def log_density(geography: str = "tract", year: int = 2022) -> tuple[dict, dict]:
    """({geoid: winsorized log10 people/km2}, {geoid: population}). Zero-land/zero-pop areas dropped."""
    gaz = _gazetteer(geography, year).set_index("geoid")["aland_km2"]
    pop = _population(geography, year)
    df = pd.DataFrame({"aland": gaz, "pop": pop}).dropna()
    df = df[(df["aland"] > 0) & (df["pop"] > 0)]
    ld = np.log10(df["pop"] / df["aland"])
    lo, hi = _weighted_quantiles(ld.to_numpy(), df["pop"].to_numpy(), [0.01, 0.99])
    return ld.clip(lo, hi).to_dict(), df["pop"].to_dict()


def _weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w) / w.sum()
    return [float(v[np.searchsorted(cum, q)]) for q in qs]


def belt_anchors(gss_binned, log_dens: dict, pop: dict) -> dict:
    """{srcbelt code: anchor log10 density} by share-matching GSS belt shares onto the national
    pop-weighted tract density distribution (see module docstring)."""
    d = gss_binned.dropna(subset=["srcbelt"]).copy()
    w = d["wtssps"].fillna(1.0) if "wtssps" in d else pd.Series(1.0, index=d.index)
    shares = {b: float(w[d["srcbelt"] == b].sum() / w.sum()) for b in BELT_URBAN_ORDER}
    geoids = list(log_dens)
    ld = np.array([log_dens[g] for g in geoids])
    pw = np.array([pop[g] for g in geoids])
    anchors, cum = {}, 0.0
    for b in BELT_URBAN_ORDER:
        mid = cum + shares[b] / 2
        anchors[b] = _weighted_quantiles(ld, pw, [min(mid, 1.0)])[0]
        cum += shares[b]
    return anchors
