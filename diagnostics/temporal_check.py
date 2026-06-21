"""Can a temporal panel PREDICT the national happiness trend? Honest answer via leave-one-year-out.

A period spline fits the trend beautifully in-sample, but leave-one-year-out shows it overfits: held out,
it can't predict a year's mood. National happiness mood is idiosyncratic period shock, not a smooth
predictable trend — so a temporal panel can carry per-year *observed* national levels (calibration), but
must NOT pretend to model/forecast the mood. Renders docs/temporal_overfit.png + prints the verdict.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from microhappiness.binning import bin_gss  # noqa: E402
from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors  # noqa: E402
from microhappiness import temporal  # noqa: E402

INK, ACTUAL, FIT, OOS = "#1f2a2b", "#c2562a", "#2b6a6c", "#9b59b6"


def main():
    gss = bin_gss(recode_predictors(load_gss("data/gss_cumulative.dta", columns=GSS_COLUMNS)))
    m = temporal.fit_temporal(gss)
    ins = temporal.per_year_rates(gss, m)
    loyo = temporal.leave_one_year_out(gss).dropna()
    r_ins = ins.actual.corr(ins.modeled)
    r_oos = loyo.actual.corr(loyo.pred_oos)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(2020.4, 2024.6, color="#999", alpha=0.10)
    ax.plot(ins.year, ins.actual, "-o", color=ACTUAL, lw=2.2, ms=4.5, label="Actual GSS")
    ax.plot(ins.year, ins.modeled, "-", color=FIT, lw=2.2, label=f"Period spline, in-sample (r={r_ins:.2f})")
    ax.plot(loyo.year, loyo.pred_oos, "o--", color=OOS, lw=1.4, ms=5, mfc="white",
            label=f"…held out (leave-one-year-out, r={r_oos:.2f})")
    ax.set_title("There's no honest temporal forecast: the period model overfits", color=INK,
                 fontsize=12.5, weight="bold", pad=24)
    ax.text(0.5, 1.025, "in-sample the spline hugs the trend; held out it can't predict a year's mood — "
            "national mood is period shock, not a smooth trend", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.5, color="#666")
    ax.set_ylabel("% very happy", color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/temporal_overfit.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"in-sample r={r_ins:.3f} | leave-one-year-out r={r_oos:.3f} "
          f"(in-person only r={loyo[loyo.web==0].actual.corr(loyo[loyo.web==0].pred_oos):.3f})")
    print("VERDICT: the national mood trend is not predictable from a smooth period model — a temporal")
    print("panel must carry OBSERVED per-year national levels (calibration), not a forecast. -> docs/temporal_overfit.png")


if __name__ == "__main__":
    main()
