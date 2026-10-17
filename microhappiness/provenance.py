"""What a run was built from — recorded in aggregation_spec.json so two runs can be told apart.

Small value differences between machines come from three places: library versions, data releases that
changed under the same URL/ID (PLACES rows, the "latest" GSS file), and code. This captures the first
two; the git commit captures the third.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.request import urlopen

PACKAGES = ("pandas", "numpy", "scipy", "statsmodels", "patsy")


def environment() -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {"python": platform.python_version(), "packages": versions, "git_commit": _git_commit()}


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parent.parent
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "microhappiness"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def gss_file(path, gss) -> dict:
    """Fingerprint of the GSS cumulative file + the survey years actually present (with HAPPY asked)."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    years = gss.loc[gss["happy"].notna(), "year"]
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest(),
            "years": gss_years(gss), "n_years": int(years.nunique())}


def gss_years(gss) -> str:
    """'<first>-<last>' survey years with a HAPPY answer, read from the data rather than hard-coded."""
    years = gss.loc[gss["happy"].notna(), "year"]
    return f"{int(years.min())}-{int(years.max())}"


def socrata_release(resource_url: str) -> dict:
    """Release metadata for a data.cdc.gov resource: dataset id, name, and when its rows last changed."""
    dataset_id = resource_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".json")
    out = {"dataset_id": dataset_id}
    try:
        meta = json.loads(urlopen(f"https://data.cdc.gov/api/views/{dataset_id}.json", timeout=60).read())
    except Exception as exc:  # metadata is informative only; never fail a run over it
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["name"] = meta.get("name")
    if meta.get("rowsUpdatedAt"):
        out["rows_updated_at"] = datetime.fromtimestamp(meta["rowsUpdatedAt"], timezone.utc).isoformat()
    return out
