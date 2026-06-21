"""Validate the modeled estimates. Three axes, honest about what each can and can't show.

1. NATIONAL LEVEL — the modeled population-weighted national rate vs the design-weighted GSS rate.
   (After calibration these match by construction; pre-calibration this is the gap calibration closes.)
2. CONVERGENT VALIDITY — correlate modeled tract happiness against an independent free outcome that is
   NOT a model input but is wellbeing-linked: CDC tract life expectancy (USALEEP). A positive
   correlation is supportive, not proof — life expectancy still shares causes with our inputs.
3. GOLD STANDARD (not here yet) — (a) modeled national trend vs the actual GSS HAPPY trend by year needs
   the temporal panel; (b) county aggregates vs the Sharecare/Gallup Community Well-Being Index needs
   that licensed data. Both are tracked in METHODOLOGY_TODO.md.
"""

from __future__ import annotations

import csv
import io
from urllib.request import urlopen

import numpy as np
import pandas as pd

# CDC NCHS US Small-area Life Expectancy Estimates Project — tract life expectancy (2010-2015), public.
USALEEP_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NVSS/USALEEP/US_A.CSV"


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
    """{tract_geoid (11-digit): life_expectancy_years} from CDC USALEEP."""
    text = urlopen(USALEEP_URL, timeout=300).read().decode("utf-8", "replace")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        geoid, le = row.get("Tract ID"), row.get("e(0)")
        if geoid and le not in (None, ""):
            try:
                out[str(geoid).zfill(11)] = float(le)
            except ValueError:
                continue
    return out


def convergent_validity(modeled_tract: pd.DataFrame, life_exp: dict) -> dict:
    """Correlate modeled tract happiness with independent tract life expectancy."""
    df = modeled_tract.copy()
    df["geoid"] = df["geoid"].astype(str).str.zfill(11)
    df["life_exp"] = df["geoid"].map(life_exp)
    paired = df.dropna(subset=["life_exp", "happiness_index"])
    return {
        "n_tracts_paired": int(len(paired)),
        "pearson_index_vs_life_expectancy": round(float(paired["happiness_index"].corr(paired["life_exp"])), 3),
        "pearson_pct_very_vs_life_expectancy": round(float(paired["pct_very_happy"].corr(paired["life_exp"])), 3),
        "note": "Convergent validity, not independent proof: life expectancy shares causes (health, income) "
                "with the model's inputs. A strong positive correlation supports face validity.",
    }
