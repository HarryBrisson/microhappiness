"""GSS microdata: load the cumulative datafile and recode predictors to match ACS categories.

The recodes are the contract between survey and census: a GSS `marital` category must mean the same
thing as the ACS B12001 bucket it'll be poststratified against. Keep these aligned with acs.py.

Source: GSS cumulative datafile 1972-present (https://gss.norc.org/get-the-data), Stata/SPSS. Or the
`gssr` R package. Public use; geography is region-only (which is *why* we model + poststratify).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# GSS columns we pull (cumulative-file names, lowercased on load).
GSS_COLUMNS = (
    "year", "happy", "marital", "age", "sex", "race", "hispanic",
    "wrkstat", "educ", "degree", "realinc", "income", "region",
    "wtssps", "wtssall",
)

HAPPY_RECODE = {1: 3, 2: 2, 3: 1}  # GSS: 1=very,2=pretty,3=not too  ->  3/2/1 (higher = happier)


def load_gss(path: str | Path, columns: tuple[str, ...] | None = GSS_COLUMNS) -> pd.DataFrame:
    """Read the GSS cumulative .dta/.sav, lowercase columns, keep the ones we need."""
    path = Path(path)
    reader = pd.read_stata if path.suffix.lower() == ".dta" else pd.read_spss
    df = reader(path)
    df.columns = [c.lower() for c in df.columns]
    if columns:
        df = df[[c for c in columns if c in df.columns]].copy()
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
    return out


def _race_ethnicity(df: pd.DataFrame) -> pd.Series:
    hisp = df.get("hispanic", pd.Series(1, index=df.index)) != 1  # HISPANIC==1 is "not hispanic"
    race = df["race"]  # 1=white, 2=black, 3=other
    out = pd.Series("other_nh", index=df.index)
    out[race == 1] = "white_nh"
    out[race == 2] = "black_nh"
    out[hisp] = "hispanic"
    return out
