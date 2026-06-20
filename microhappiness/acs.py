"""ACS via the Census API: poststratification marginals + area covariates, tract and ZCTA.

Comprehensive on the *data* side (pull every table the candidate models might need) so the model
side can stay lean. Categories here must mirror the GSS recodes in gss.py.

Census API: https://api.census.gov/data/<year>/acs/acs5  (key required, CENSUS_API_KEY).
"""

from __future__ import annotations

import os

# Table IDs -> what they supply. Pull broadly; each candidate model uses a subset.
ACS_TABLES = {
    "B12001": "marital status by sex",
    "B19001": "household income (16 brackets) -> income deciles",
    "B19013": "median household income (area covariate)",
    "B23025": "employment status (full-time / labor force)",
    "B01001": "age x sex (poststrat backbone)",
    "B15003": "educational attainment (25+)",
    "B03002": "race x hispanic origin",
    "C18120": "disability x employment (optional predictor)",
    "B11001": "household type (composition, optional)",
}

# Area-level covariates for the PLACES-analog model (joined by GEOID, not poststratified).
AREA_COVARIATES = {
    "pct_married": ("B12001", "married-of-15+"),
    "median_income_z": ("B19013", "z-scored within nation/year"),
}

GEOGRAPHIES = ("tract", "zcta")


def census_key() -> str:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("Set CENSUS_API_KEY (https://api.census.gov/data/key_signup.html)")
    return key


def fetch_marginals(year: int, geography: str, tables: tuple[str, ...] = tuple(ACS_TABLES)) -> "pd.DataFrame":
    """Return per-area marginal counts for each requested table, keyed by GEOID.

    Implementation plan: page the Census API per table (it caps variables/call), recode raw
    variable columns into the model categories (mirror gss.recode_predictors), and return a tidy
    {GEOID, dimension, category, count} frame that poststratify.py rakes into a joint table.
    For ZCTA the predicate is `for=zip code tabulation area:*`; for tract it's `for=tract:*&in=state:..`.
    """
    raise NotImplementedError("fetch_marginals: page Census API per table, recode to model categories")


def fetch_area_covariates(year: int, geography: str) -> "pd.DataFrame":
    """Return {GEOID: {pct_married, median_income_z, ...}} for the PLACES-analog area terms."""
    raise NotImplementedError("fetch_area_covariates: B12001/B19013 -> covariates")


def fetch_census_joint_cells(year: int, geography: str) -> "pd.DataFrame":
    """Census joint counts for age x sex x race x education (the M2/PLACES poststrat cells).

    These come closest to a published *joint* distribution; M2 uses them directly and avoids
    synthesizing a joint via raking. Returns {GEOID, age_bucket, sex, race_ethnicity, education, count}.
    """
    raise NotImplementedError("fetch_census_joint_cells: published joint age/sex/race/edu cells")
