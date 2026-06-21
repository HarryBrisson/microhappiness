"""Render the README visuals: the national-trend validation chart + a county choropleth of estimates.

  python -m diagnostics.make_visuals    # writes docs/national_trend.png + docs/happiness_map.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from microhappiness.binning import bin_gss  # noqa: E402
from microhappiness.estimate import fit_models  # noqa: E402
from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors  # noqa: E402
from microhappiness.validate import national_trend  # noqa: E402

INK, ACCENT, TEAL = "#1f2a2b", "#c2562a", "#2b6a6c"
COUNTIES = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


MODE_BREAK = (2021, 2024)  # GSS switched to web/push-to-web — a documented survey-mode artifact


def trend_chart(gss, out: Path):
    logit, _ols, _n = fit_models(gss)
    df, r_all = national_trend(gss, logit)
    keep = df[~df.year.isin(MODE_BREAK)]
    r_ex = round(float(keep.actual.corr(keep.modeled)), 2)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(2020.4, 2024.6, color="#999", alpha=0.10)
    ax.plot(df.year, df.actual, "-o", color=ACCENT, lw=2.2, ms=4.5, label="Actual GSS")
    ax.plot(df.year, df.modeled, "-o", color=TEAL, lw=2.2, ms=4.5, label="Modeled (composition + health)")
    ax.annotate("2021 / 2024: GSS web-mode change\n(a survey artifact, not a real crash)",
                xy=(2021, 19.6), xytext=(2006, 21.5), fontsize=8.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#999", lw=1))
    ax.set_title("National happiness: the model tracks geography, not period shocks", color=INK,
                 fontsize=12.5, weight="bold", pad=26)
    ax.text(0.5, 1.025, f"weak correlation over time (r = {r_ex} excluding the web-mode waves) — "
            "the temporal trend needs the panel", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#666")
    ax.set_ylabel("% very happy", color=INK)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return df, r_all, r_ex


def county_map(tract_csv, out: Path):
    import geopandas as gpd

    t = pd.read_csv(tract_csv, dtype={"geoid": str})
    t["fips"] = t.geoid.str[:5]
    t["wx"] = t.happiness_index * t.adult_pop
    cty = t.groupby("fips").agg(wx=("wx", "sum"), popw=("adult_pop", "sum"))
    cty["happiness"] = cty["wx"] / cty["popw"]
    gdf = gpd.read_file(COUNTIES)
    gdf["fips"] = gdf["STATE"].astype(str).str.zfill(2) + gdf["COUNTY"].astype(str).str.zfill(3)
    gdf = gdf.merge(cty.reset_index()[["fips", "happiness"]], on="fips", how="left")

    fig, ax = plt.subplots(figsize=(10, 6.2))
    gdf.plot(column="happiness", cmap="RdYlGn", linewidth=0, ax=ax, legend=True,
             missing_kwds={"color": "#eee"},
             vmin=cty["happiness"].quantile(.05), vmax=cty["happiness"].quantile(.95),
             legend_kwds={"shrink": 0.5, "label": "Modeled happiness index (0–100)"})
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24, 49.5)  # contiguous US
    ax.set_title("Modeled happiness by county — from circumstantial ACS + PLACES variables",
                 color=INK, fontsize=13, weight="bold")
    ax.axis("off")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return len(cty)


def main():
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    gss = bin_gss(recode_predictors(load_gss("data/gss_cumulative.dta", columns=GSS_COLUMNS)))
    df, r_all, r_ex = trend_chart(gss, docs / "national_trend.png")
    print(f"trend chart: {len(df)} years {int(df.year.min())}-{int(df.year.max())}, "
          f"r={r_all} all / {r_ex} excl mode-break -> docs/national_trend.png")
    n = county_map("data/national/happiness_tract.csv", docs / "happiness_map.png")
    print(f"county map: {n} counties -> docs/happiness_map.png")


if __name__ == "__main__":
    main()
