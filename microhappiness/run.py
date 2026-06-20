"""End-to-end pipeline: GSS model -> ACS+PLACES margins -> raked happiness estimates, any geography.

  # nationwide tracts, M5 (health + mental + smoking), published for Penlight:
  python -m microhappiness.run --geography tract --model m5 --out-dir data/national
  # nationwide ZCTAs:
  python -m microhappiness.run --geography zcta --model m5 --out-dir data/national
  # one state (debug):
  python -m microhappiness.run --geography tract --states 17

Fits the model + seed ONCE, fetches PLACES nationally, then per geography pulls ACS margins (tracts
loop the states; ZCTAs are one national query), rakes the seed onto each area, and writes
happiness_<geo>.csv + (with --out-dir) the byop/v1 aggregation_spec.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from microhappiness.acs import fetch_acs_margins
from microhappiness.binning import bin_gss
from microhappiness.estimate import estimate_state, estimate_state_m5, fit_m5
from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors
from microhappiness.places import fetch_measure
from microhappiness.publish import write_spec

# 50 states + DC: FIPS -> USPS abbr.
STATES = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geography", choices=["tract", "zcta"], default="tract")
    ap.add_argument("--model", choices=["m4", "m5"], default="m5")
    ap.add_argument("--states", default="all", help="'all' or comma-separated FIPS (tract only)")
    ap.add_argument("--gss", default="data/gss_cumulative.dta")
    ap.add_argument("--acs-year", type=int, default=2022)
    ap.add_argument("--out-dir", default="data/national")
    args = ap.parse_args()
    geo = args.geography

    print("fitting GSS model + seed …")
    gss = bin_gss(recode_predictors(load_gss(args.gss, columns=GSS_COLUMNS)))
    measures = ("GHLTH", "MHLTH", "CSMOKING") if args.model == "m5" else ("GHLTH",)
    print(f"fetching PLACES {geo} margins ({', '.join(measures)}) …")
    places = {m: fetch_measure(m, geography=geo) for m in measures}
    fitted = fit_m5(gss) if args.model == "m5" else None

    def estimate(acs):
        if args.model == "m5":
            return estimate_state_m5(gss, acs, places, fitted=fitted)
        return estimate_state(gss, acs, places["GHLTH"], fitted=fitted)

    parts = []
    if geo == "zcta":
        acs = fetch_acs_margins(geography="zcta", year=args.acs_year)
        df, meta = estimate(acs)
        parts.append(df)
    else:
        fips = list(STATES) if args.states == "all" else args.states.split(",")
        for i, st in enumerate(fips, 1):
            acs = fetch_acs_margins(st, year=args.acs_year)
            df, meta = estimate(acs)
            parts.append(df)
            print(f"  [{i}/{len(fips)}] {STATES.get(st, st)}: {len(df)} {geo}s")

    full = pd.concat(parts, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"happiness_{geo}.csv"
    full.to_csv(out, index=False)
    write_spec(out_dir, [geo], vintage=str(args.acs_year), model_key=args.model,
               gss_years="1972-2022")
    s = full.sort_values("happiness_index")
    print(f"\nmodel {args.model} fit on N={meta['n_fit']} / {meta['n_cells']} cells; "
          f"{len(full)} {geo}s -> {out}")
    print(f"happiness_index: median {full['happiness_index'].median():.1f}, "
          f"range {full['happiness_index'].min():.1f}–{full['happiness_index'].max():.1f}; "
          f"pct_very_happy median {full['pct_very_happy'].median():.1f}%")


if __name__ == "__main__":
    main()
