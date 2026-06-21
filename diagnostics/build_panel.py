"""Build a compositional temporal panel for a state across two ACS vintages, on consistent 2020 tracts.

Demonstrates the two new pieces — per-year ACS + the 2010->2020 tract crosswalk — and the honest payoff:
the NATIONAL compositional drift is tiny (composition barely moves average happiness), but specific areas
with real demographic change (gentrification / decline) move several points. Renders a Cook County
"compositional change" map.

  python -m diagnostics.build_panel    # IL 2013 -> 2022, Chicago change map + change CSV
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from microhappiness.acs import fetch_acs_margins  # noqa: E402
from microhappiness.binning import bin_gss  # noqa: E402
from microhappiness.crosswalk import convert_to_2020, load_crosswalk  # noqa: E402
from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors  # noqa: E402
from microhappiness import temporal  # noqa: E402

COOK_TRACTS_2020 = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2023/"
                    "MapServer/8/query?where=STATE%3D%2717%27%20AND%20COUNTY%3D%27031%27"
                    "&outFields=GEOID&returnGeometry=true&f=geojson&outSR=4326")


def build(state="17", y0=2013, y1=2022):
    gss = bin_gss(recode_predictors(load_gss("data/gss_cumulative.dta", columns=GSS_COLUMNS)))
    fit = temporal.fit_circumstantial(gss)
    e0 = convert_to_2020(temporal.estimate_year(fetch_acs_margins(state, year=y0), fit), load_crosswalk(state))
    e1 = temporal.estimate_year(fetch_acs_margins(state, year=y1), fit)
    df = pd.DataFrame({f"y{y0}": pd.Series(e0), f"y{y1}": pd.Series(e1)}).dropna()
    df["change"] = df[f"y{y1}"] - df[f"y{y0}"]
    return df


def chicago_map(df, out: Path, y0, y1):
    import geopandas as gpd

    gj = json.loads(urlopen(COOK_TRACTS_2020, timeout=180).read())
    gdf = gpd.GeoDataFrame.from_features(gj["features"])
    gdf["change"] = gdf["GEOID"].astype(str).map(df["change"])
    fig, ax = plt.subplots(figsize=(7.5, 8))
    gdf.plot(column="change", cmap="RdBu", linewidth=0, ax=ax, legend=True, vmin=-3, vmax=3,
             missing_kwds={"color": "#eee"},
             legend_kwds={"shrink": 0.4, "label": "compositional happiness change (index pts)"})
    ax.set_title(f"Chicago: modeled-happiness change from demographic composition, {y0}→{y1}\n"
                 "(blue = composition improved; a synthetic estimate)", fontsize=11, weight="bold")
    ax.axis("off")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    df = build()
    df.to_csv("data/national/panel_IL_2013_2022.csv")
    print(f"IL panel: {len(df)} tracts on 2020 geography | national drift {df.change.mean():+.2f} pts "
          f"| per-tract sd {df.change.std():.2f} (p10..p90 {df.change.quantile(.1):+.1f}..{df.change.quantile(.9):+.1f})")
    chicago_map(df, docs / "panel_chicago_change.png", 2013, 2022)
    print("-> docs/panel_chicago_change.png")


if __name__ == "__main__":
    main()
