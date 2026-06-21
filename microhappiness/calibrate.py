"""Benchmark the modeled national level to the GSS survey rate (v2 calibration).

The poststratified estimates reproduce the geographic PATTERN well but sit a few points below the raw
GSS national rate (M5's negative mental/smoking margins pull the level down). We additively recenter so
the population-weighted national mean matches the design-weighted GSS national rate — this fixes the
absolute level while leaving the relative geography (what a map shows) untouched. The offset is recorded
in the spec for transparency.
"""

from __future__ import annotations

import numpy as np

_COLS = ("happiness_index", "pct_very_happy")


def gss_national(gss_binned) -> dict:
    """Design-weighted GSS national very-happy % and mean 0-100 index — the calibration target."""
    d = gss_binned.dropna(subset=["happy"])
    w = np.asarray(d["wtssps"].fillna(1.0) if "wtssps" in d else np.ones(len(d)), dtype=float)
    very = (d["happy"] == 3).astype(float).to_numpy()
    index = d["happy"].map({3: 100.0, 2: 50.0, 1: 0.0}).to_numpy()
    return {"pct_very_happy": float((very * w).sum() / w.sum() * 100.0),
            "happiness_index": float((index * w).sum() / w.sum())}


def offsets(df, target: dict) -> dict:
    """Additive offsets making df's population-weighted means match `target`."""
    pop = df["adult_pop"].to_numpy(dtype=float)
    pop = pop if pop.sum() > 0 else np.ones(len(df))
    return {col: round(target[col] - float(np.average(df[col], weights=pop)), 3) for col in _COLS}


def apply_offsets(df, offs: dict):
    """Return a copy of df with each offset added (and CI columns shifted in step, if present)."""
    out = df.copy()
    for col, off in offs.items():
        for suffix in ("", "_lo", "_hi"):
            if (c := col + suffix) in out:
                out[c] = out[c] + off
    return out
