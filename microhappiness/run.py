"""End-to-end v1 pipeline: GSS model -> ACS+PLACES tract margins -> raked tract happiness estimates.

  python -m microhappiness.run --state 17 --abbr IL --out data/happiness_IL.csv

Fits M4 (circumstantial + health) on the GSS cumulative file, fetches one state's tract margins from
the Census API + CDC PLACES, poststratifies (rakes) the GSS seed onto each tract, and writes a tract
table {geoid, happiness_index, pct_very_happy, adult_pop}. Nationwide = loop the states (v2 also adds
M5 mental/smoking via imputation, Monte-Carlo CIs, ZCTA, and the byop/v1 publish spec).
"""

from __future__ import annotations

import argparse

from microhappiness.acs import fetch_acs_margins
from microhappiness.binning import bin_gss
from microhappiness.estimate import estimate_state
from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors
from microhappiness.places import fetch_measure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True, help="2-digit state FIPS, e.g. 17")
    ap.add_argument("--abbr", required=True, help="2-letter state abbr for PLACES, e.g. IL")
    ap.add_argument("--gss", default="data/gss_cumulative.dta")
    ap.add_argument("--acs-year", type=int, default=2022)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("fitting GSS model + seed …")
    gss = bin_gss(recode_predictors(load_gss(args.gss, columns=GSS_COLUMNS)))
    print(f"fetching ACS {args.acs_year} + PLACES margins for {args.abbr} …")
    acs = fetch_acs_margins(args.state, year=args.acs_year)
    health = fetch_measure(args.abbr, "GHLTH")
    df, meta = estimate_state(gss, acs, health)

    out = args.out or f"data/happiness_{args.abbr}.csv"
    df.to_csv(out, index=False)
    df = df.sort_values("happiness_index")
    print(f"\nmodel fit on N={meta['n_fit']} GSS rows over {meta['n_cells']} cells; "
          f"estimated {len(df)} tracts -> {out}")
    print(f"happiness_index: median {df['happiness_index'].median():.1f}, "
          f"range {df['happiness_index'].min():.1f}–{df['happiness_index'].max():.1f}; "
          f"pct_very_happy median {df['pct_very_happy'].median():.1f}%")
    print("  lowest 3 tracts: ", list(df.head(3)["geoid"]))
    print("  highest 3 tracts:", list(df.tail(3)["geoid"]))


if __name__ == "__main__":
    main()
