"""Optional per-stage dumps (`--checkpoint-dir`) so a replication can be compared stage by stage.

Each file is small and plain (CSV/JSON), named by stage number, so diffing two runs' directories
shows the first stage where they part ways: the GSS sample, the fitted coefficients, the design
columns, the seed-cell predictions, raking convergence, then the published values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class Checkpoints:
    """Writes stage files under `root`; every method is a no-op when root is None."""

    def __init__(self, root):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def __bool__(self):
        return self.root is not None

    def json(self, name: str, obj) -> None:
        if self.root:
            (self.root / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_jsonable))

    def csv(self, name: str, df: pd.DataFrame) -> None:
        if self.root:
            df.to_csv(self.root / f"{name}.csv", index=False)


def _jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def gss_sample(gss, n_fit: int) -> dict:
    happy = gss["happy"].notna()
    w = gss.loc[happy, "wtssps"] if "wtssps" in gss else None
    return {"rows": len(gss), "rows_with_happy": int(happy.sum()), "n_fit_complete_case": n_fit,
            "years_with_happy": sorted(int(y) for y in gss.loc[happy, "year"].unique()),
            "wtssps_sum": None if w is None else float(w.fillna(0).sum()),
            "share_very_happy_unweighted": float((gss.loc[happy, "happy"] == 3).mean())}


def coefficients(fits: dict) -> pd.DataFrame:
    """One row per (model, term): estimate, SE, and the model's fit stats repeated for convenience."""
    rows = []
    for model_name, res in fits.items():
        fit_stat = getattr(res, "prsquared", None)
        if fit_stat is None:
            fit_stat = getattr(res, "rsquared", None)
        for term in res.params.index:
            rows.append({"model": model_name, "term": term, "estimate": float(res.params[term]),
                         "se": float(res.bse[term]), "nobs": int(res.nobs),
                         "fit_r2_or_pseudo_r2": None if fit_stat is None else float(fit_stat)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, columns) -> dict:
    out = {"n_areas": len(df)}
    for c in columns:
        if c in df:
            s = df[c]
            out[c] = {"mean": float(s.mean()), "p10": float(s.quantile(0.1)), "median": float(s.median()),
                      "p90": float(s.quantile(0.9)), "min": float(s.min()), "max": float(s.max())}
    return out
