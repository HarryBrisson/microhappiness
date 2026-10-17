"""Rebuild a fitted formula model's design matrix on new rows (the seed cells), with a guard.

The CI draws multiply a seed-cell design matrix by drawn coefficients, so that matrix must line up
column-for-column with what statsmodels actually fit. statsmodels has kept `design_info` in different
places across releases (`model.data.design_info`, `model.design_info`, and in 0.15
`model.data.model_spec`; statsmodels #7043), and a missing one has been patched in the past by
re-running patsy by hand. A rebuilt design_info is only safe if it reproduces the fit-time design
exactly, so `seed_design` checks that before returning.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class DesignMismatch(RuntimeError):
    """The design matrix we can rebuild does not match the one the model was fit on."""


def design_info_of(result):
    """The patsy DesignInfo a formula-fit result was built with (rebuilt from the fit data if absent).

    statsmodels <=0.14 keeps it at `model.data.design_info`; 0.15 moved it to `model.data.model_spec`
    (found by Takashi Nishikawa, PR #1), which holds a formulaic ModelSpec instead if that engine is used.
    """
    import patsy

    model = result.model
    for owner in (model.data, model):
        for attr in ("design_info", "model_spec"):
            info = getattr(owner, attr, None)
            if isinstance(info, patsy.DesignInfo):
                return info
    formula = getattr(model, "formula", None)
    frame = getattr(model.data, "frame", None)
    if formula is None or frame is None:
        raise DesignMismatch("model has no design_info and no formula/frame to rebuild it from "
                             "(was it fit from arrays, or saved with remove_data=True?)")
    log.warning("design_info missing on %s; rebuilding it from the formula with patsy",
                type(model).__name__)
    return patsy.dmatrices(formula, frame, return_type="dataframe")[1].design_info


def check_design(result, info) -> None:
    """Raise DesignMismatch unless `info` reproduces the model's fit-time exog exactly."""
    import patsy

    model = result.model
    if list(info.column_names) != list(result.params.index):
        raise DesignMismatch(f"design columns {info.column_names} != params {list(result.params.index)}")
    frame = getattr(model.data, "frame", None)
    if frame is None:
        return
    rows = model.data.row_labels
    rebuilt = np.asarray(patsy.build_design_matrices([info], frame.loc[rows])[0])
    if rebuilt.shape != model.exog.shape or not np.allclose(rebuilt, model.exog, equal_nan=True):
        raise DesignMismatch("rebuilt design matrix differs from the matrix the model was fit on")


def seed_design(result, seed) -> np.ndarray:
    """Seed-cell design matrix for `result`, guaranteed to line up with `result.params`."""
    import patsy

    info = design_info_of(result)
    check_design(result, info)
    return np.asarray(patsy.build_design_matrices([info], seed)[0])
