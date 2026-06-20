"""Step 1 — broad empirical screen on GSS's own variables, then a theory gate.

Discovery, NOT the final model. The exploratory pass can only run where individual happiness and the
candidate predictor sit together — that's GSS itself (hundreds of items), not ACS/PLACES (no fine
geography to attach to a GSS respondent). So we screen GSS-wide to see what actually predicts
happiness in-sample, then keep only variables that are simultaneously:
  (a) empirically useful (incremental, cross-validated signal — not noise),
  (b) reproducible at tract/ZCTA via ACS or PLACES, and
  (c) backed by a clear, logical mechanism (a HUMAN call — guards against spurious correlations).

This re-ranks predictors against the data instead of trusting the literature, while the theory gate
keeps the final model parsimonious (the research warned explicitly against kitchen-sink overfitting).

Guards against a broad screen chasing noise: a held-out split + cross-validated incremental
pseudo-R², and reporting effect stability across GSS eras (a real predictor is stable; a fluke isn't).
"""

from __future__ import annotations

# Local availability of a discovered construct — the (b) filter. Hand-maintained as the screen runs.
LOCAL_SOURCE = {
    "marital": "ACS B12001",
    "income": "ACS B19001",
    "employment": "ACS B23025",
    "education": "ACS B15003",
    "age": "ACS B01001",
    "sex": "ACS B01001",
    "race_ethnicity": "ACS B03002",
    "health": "CDC PLACES GHLTH (fair-or-poor)",   # the PLACES-unlocked margin
    # constructs GSS shows matter but ACS/PLACES can't supply -> cannot poststratify (document, exclude):
    "attend": None,      # religious attendance
    "trust": None,       # social trust
    "polviews": None,    # political ideology
    "satfin": None,      # financial satisfaction (subjective; partly proxied by income)
}


def screen(gss_df, candidate_vars, *, holdout: float = 0.3, folds: int = 5):
    """Rank candidate GSS variables by cross-validated incremental signal for HAPPY.

    Returns rows: {var, univariate_assoc, cv_incremental_pseudo_r2, era_stability, local_source,
    mechanism_note}. local_source from LOCAL_SOURCE; mechanism_note is filled by a human for the (c)
    gate. Only var with cv signal AND a non-null local_source AND an accepted mechanism advance to a
    candidate model in models.py.
    """
    raise NotImplementedError("univariate + CV-incremental screen with held-out split; flag availability")
