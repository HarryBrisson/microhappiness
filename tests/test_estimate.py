"""Smoke test of the fit -> seed design matrix -> prediction path on a synthetic GSS-like frame.

Runs in seconds with no downloads. It pins down the invariant the CI draws depend on: the seed-cell
design matrix rebuilt from the fit lines up with the coefficients, and reproduces statsmodels' own
predictions. If a library upgrade moves `design_info` again, this fails before a nationwide run does.
"""

import numpy as np
import pandas as pd
import pytest

from microhappiness import estimate
from microhappiness._design import DesignMismatch, check_design, design_info_of, seed_design


@pytest.fixture(scope="module")
def gss():
    rng = np.random.default_rng(7)
    n = 4000
    d = pd.DataFrame({
        "married": rng.integers(0, 2, n).astype(float),
        "employment": rng.choice(["employed", "unemployed", "nilf"], n, p=[0.6, 0.1, 0.3]),
        "home_owner": rng.integers(0, 2, n).astype(float),
        "lives_alone": rng.integers(0, 2, n).astype(float),
        "income4": rng.integers(0, 4, n).astype(float),
        "health": rng.integers(0, 2, n).astype(float),
        "mental_health": rng.integers(0, 2, n).astype(float),
        "smoker": rng.integers(0, 2, n).astype(float),
        "wtssps": rng.uniform(0.5, 2.0, n),
    })
    lin = -1.2 + 0.5 * d["married"] + 0.4 * d["health"] - 0.6 * (d["employment"] == "unemployed")
    p3 = 1 / (1 + np.exp(-lin))
    d["happy"] = np.where(rng.uniform(size=n) < p3, 3, rng.choice([1, 2], n, p=[0.2, 0.8])).astype(float)
    d.loc[rng.choice(n, 200, replace=False), "health"] = np.nan  # complete-case dropping must still line up
    return d


def test_seed_design_matches_statsmodels_predictions(gss):
    logit, ols, n_fit = estimate.fit_models(gss)
    assert n_fit == gss.dropna(subset=["health"]).shape[0]
    seed = estimate.build_seed(gss)
    X = seed_design(logit, seed)
    assert X.shape == (len(seed), len(logit.params))
    np.testing.assert_allclose(X @ logit.params.to_numpy(), logit.predict(seed, which="linear"))
    np.testing.assert_allclose(X @ ols.params.to_numpy(), ols.predict(seed))


def test_rebuilt_design_info_is_checked(gss):
    import statsmodels.formula.api as smf

    logit, _, _ = estimate.fit_models(gss)
    check_design(logit, design_info_of(logit))  # the real one passes
    fewer = smf.logit("vh ~ married + health", data=gss.assign(vh=(gss["happy"] == 3).astype(int))).fit(disp=0)
    with pytest.raises(DesignMismatch, match="columns"):
        check_design(logit, design_info_of(fewer))
    # Same column names, different values: a stateful transform learned on other data.
    a = pd.DataFrame({"y": [1.0, 2.0, 3.0, 5.0], "x": [1.0, 2.0, 3.0, 4.0]})
    fit_a = smf.ols("y ~ center(x)", data=a).fit()
    fit_b = smf.ols("y ~ center(x)", data=a.assign(x=a["x"] + 10)).fit()
    with pytest.raises(DesignMismatch, match="differs"):
        check_design(fit_a, design_info_of(fit_b))


def test_missing_design_info_is_rebuilt_and_verified(gss):
    logit, _, _ = estimate.fit_models(gss)
    seed = estimate.build_seed(gss)
    expected = seed_design(logit, seed)
    # Hide every place statsmodels keeps it (0.14: design_info, 0.15: model_spec) to force the rebuild.
    data = logit.model.data
    saved = {a: data.__dict__.pop(a) for a in ("design_info", "model_spec") if a in data.__dict__}
    try:
        np.testing.assert_allclose(seed_design(logit, seed), expected)
    finally:
        data.__dict__.update(saved)


def test_design_info_found_where_this_statsmodels_keeps_it(gss, caplog):
    logit, _, _ = estimate.fit_models(gss)
    with caplog.at_level("WARNING", logger="microhappiness._design"):
        design_info_of(logit)
    assert "rebuilding" not in caplog.text  # found in place (design_info or model_spec), not rebuilt


def test_m5_estimates_have_cis_and_schema(gss):
    fitted = estimate.fit_m5(gss, n_draws=50)
    margin = {
        "married": {1.0: 0.5, 0.0: 0.5},
        "employment": {"employed": 0.6, "unemployed": 0.1, "nilf": 0.3},
        "home_owner": {1.0: 0.6, 0.0: 0.4},
        "lives_alone": {1.0: 0.3, 0.0: 0.7},
        "income4": {0.0: 0.25, 1.0: 0.25, 2.0: 0.25, 3.0: 0.25},
    }
    places = {k: {"17031000100": {"fraction": f, "adult_pop": 1000}}
              for k, f in (("GHLTH", 0.15), ("MHLTH", 0.17), ("CSMOKING", 0.12))}
    out, meta = estimate.estimate_state_m5(gss, {"17031000100": margin}, places, fitted=fitted)
    assert list(out.columns) == ["geoid", "happiness_index", "happiness_index_lo", "happiness_index_hi",
                                 "pct_very_happy", "pct_very_happy_lo", "pct_very_happy_hi", "adult_pop"]
    row = out.iloc[0]
    assert row.pct_very_happy_lo <= row.pct_very_happy <= row.pct_very_happy_hi
    assert 0 < row.pct_very_happy < 100
