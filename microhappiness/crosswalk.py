"""2010 -> 2020 census-tract crosswalk for the temporal panel.

ACS 5-year before 2020 uses 2010 tract boundaries; 2020+ uses 2020 boundaries. To put a per-year panel on
a single consistent 2020-tract geography, we area-weight each 2010 tract's value onto the 2020 tracts it
overlaps, using the Census 2020<->2010 tract relationship file (`AREALAND_PART` = overlap land area).

Area-weighting assumes uniform density within a tract — a v1 approximation; population weighting would be
better but the relationship file only carries land area. Most tracts are unchanged (1:1) anyway.
"""

from __future__ import annotations

import csv
import io
from urllib.request import urlopen

_NATL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/tab20_tract20_tract10_natl.txt"
_STATE = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/tab20_tract20_tract10_st{st}.txt"


def load_crosswalk(state: str | None = None) -> dict:
    """{geoid20: [(geoid10, overlap_area), ...]} from the Census 2020<->2010 tract relationship file."""
    url = _STATE.format(st=str(state).zfill(2)) if state else _NATL
    text = urlopen(url, timeout=400).read().decode("utf-8-sig", "replace")
    xw: dict = {}
    for row in csv.DictReader(io.StringIO(text), delimiter="|"):
        g20, g10 = row.get("GEOID_TRACT_20"), row.get("GEOID_TRACT_10")
        try:
            area = float(row.get("AREALAND_PART") or 0)
        except ValueError:
            area = 0.0
        if g20 and g10 and area > 0:
            xw.setdefault(g20, []).append((g10, area))
    return xw


def convert_to_2020(values_2010: dict, crosswalk: dict) -> dict:
    """Area-weight a {geoid10: value} mapping onto 2020 tracts -> {geoid20: value}."""
    out = {}
    for g20, parts in crosswalk.items():
        numer = denom = 0.0
        for g10, area in parts:
            value = values_2010.get(g10)
            if value is not None:
                numer += value * area
                denom += area
        if denom > 0:
            out[g20] = numer / denom
    return out
