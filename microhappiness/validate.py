"""Validate the modeled estimates. Three axes, honest about what each can and can't show.

1. NATIONAL LEVEL — the modeled population-weighted national rate vs the design-weighted GSS rate.
   (After calibration these match by construction; pre-calibration this is the gap calibration closes.)
2. CONVERGENT VALIDITY — correlate modeled tract happiness against an independent measure RESERVED from
   the model: CDC PLACES diagnosed depression (never a predictor; we used frequent-mental-distress, a
   distinct measure). Expect a negative correlation. Supportive, not proof. (CDC tract life expectancy /
   USALEEP would be a more independent benchmark, but its public SODA copy lacks a clean tract GEOID —
   it needs the original flat file; tracked in METHODOLOGY_TODO.md.)
3. GOLD STANDARD (not here yet) — (a) modeled national trend vs the actual GSS HAPPY trend by year needs
   the temporal panel; (b) county aggregates vs the Sharecare/Gallup Community Well-Being Index needs
   that licensed data. Both are tracked in METHODOLOGY_TODO.md.
"""

from __future__ import annotations

import json
from urllib.request import urlopen

import numpy as np
import pandas as pd

# CDC NCHS US Small-area Life Expectancy Estimates (USALEEP), tract life expectancy 2010-2015, public.
USALEEP_RESOURCE = "https://data.cdc.gov/resource/5h56-n989.json"  # full_ct_num (tract GEOID) + le


def national_level(modeled: pd.DataFrame, gss_target: dict) -> dict:
    """{metric: {modeled_popweighted, gss, gap}} — the national benchmark check."""
    pop = modeled["adult_pop"].to_numpy(dtype=float)
    pop = pop if pop.sum() > 0 else np.ones(len(modeled))
    out = {}
    for col, tgt in gss_target.items():
        m = float(np.average(modeled[col], weights=pop))
        out[col] = {"modeled": round(m, 2), "gss": round(tgt, 2), "gap": round(m - tgt, 2)}
    return out


def load_tract_life_expectancy() -> dict:
    """{tract_geoid (11-digit): life_expectancy_years} from CDC USALEEP (SODA)."""
    out, offset, page = {}, 0, 50000
    while True:
        url = f"{USALEEP_RESOURCE}?$select=full_ct_num,le&$limit={page}&$offset={offset}"
        rows = json.loads(urlopen(url, timeout=300).read())
        for r in rows:
            geoid, le = r.get("full_ct_num"), r.get("le")
            if geoid and le not in (None, "", "(blank)"):
                try:
                    out[str(geoid).zfill(11)] = float(le)
                except ValueError:
                    continue
        if len(rows) < page:
            return out
        offset += page


def convergent_validity_depression(modeled_tract: pd.DataFrame, depression: dict) -> dict:
    """Correlate modeled tract happiness with CDC PLACES diagnosed depression (reserved from the model).

    `depression` = {geoid: {'fraction': ...}} from places.fetch_measure('DEPRESSION', geography='tract').
    Depression was deliberately never a predictor (we used frequent-mental-distress, a distinct measure),
    so this is a semi-independent check. Expect a negative correlation.
    """
    df = modeled_tract.copy()
    df["depression"] = df["geoid"].astype(str).map(lambda g: (depression.get(g) or {}).get("fraction"))
    paired = df.dropna(subset=["depression", "happiness_index"])
    return {
        "n_tracts_paired": int(len(paired)),
        "pearson_index_vs_depression": round(float(paired["happiness_index"].corr(paired["depression"])), 3),
        "pearson_pct_very_vs_depression": round(float(paired["pct_very_happy"].corr(paired["depression"])), 3),
        "note": "Convergent validity, not proof: depression correlates with the model's mental-distress "
                "input. A clear negative correlation supports face validity at national scale.",
    }


def convergent_validity(modeled_tract: pd.DataFrame, life_exp: dict) -> dict:
    """Correlate modeled tract happiness with independent tract life expectancy (needs a clean GEOID->LE)."""
    df = modeled_tract.copy()
    df["geoid"] = df["geoid"].astype(str).str.zfill(11)
    df["life_exp"] = df["geoid"].map(life_exp)
    paired = df.dropna(subset=["life_exp", "happiness_index"])
    return {
        "n_tracts_paired": int(len(paired)),
        "pearson_index_vs_life_expectancy": round(float(paired["happiness_index"].corr(paired["life_exp"])), 3),
        "pearson_pct_very_vs_life_expectancy": round(float(paired["pct_very_happy"].corr(paired["life_exp"])), 3),
        "note": "Convergent validity, not independent proof: life expectancy shares causes (health, income) "
                "with the model's inputs. A positive correlation supports face validity.",
    }
