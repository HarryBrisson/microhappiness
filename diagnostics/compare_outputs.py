"""Compare two runs' output CSVs (e.g. a collaborator's replication vs ours), joined on geoid.

  python -m diagnostics.compare_outputs theirs/happiness_tract.csv data/national/happiness_tract.csv
  python -m diagnostics.compare_outputs A.csv B.csv --states 17      # restrict to tract/ZCTA prefixes
  python -m diagnostics.compare_outputs A.csv B.csv --top 20          # list the biggest movers

Reports, per shared value column: mean / max absolute difference, correlation, and (where both files
carry _lo/_hi columns) how many areas' 90% intervals don't overlap. Then the geoids found in only one
file. Small, uniform shifts with r≈1 point to calibration or library drift; scattered big movers point
to data (a PLACES/ACS row that changed) — check those geoids' inputs in the --checkpoint-dir dumps.
"""

from __future__ import annotations

import argparse

import pandas as pd


def load(path: str, prefixes) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"geoid": str})
    if prefixes:
        df = df[df["geoid"].str.startswith(tuple(prefixes))]
    return df.set_index("geoid")


def compare(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    shared = a.index.intersection(b.index)
    value_cols = [c for c in a.columns if c in b.columns and not c.endswith(("_lo", "_hi"))
                  and pd.api.types.is_numeric_dtype(a[c])]
    rows = []
    for c in value_cols:
        x, y = a.loc[shared, c], b.loc[shared, c]
        d = (x - y).abs()
        row = {"column": c, "n": len(shared), "mean_abs_diff": d.mean(), "max_abs_diff": d.max(),
               "mean_signed_diff_a_minus_b": (x - y).mean(), "corr": x.corr(y)}
        lo, hi = f"{c}_lo", f"{c}_hi"
        if all(k in a.columns and k in b.columns for k in (lo, hi)):
            apart = (a.loc[shared, hi] < b.loc[shared, lo]) | (b.loc[shared, hi] < a.loc[shared, lo])
            row["ci_non_overlap"] = int(apart.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--states", default=None, help="comma-separated geoid prefixes to keep (e.g. 17)")
    ap.add_argument("--top", type=int, default=10, help="list this many biggest movers per column")
    args = ap.parse_args()
    prefixes = args.states.split(",") if args.states else None
    a, b = load(args.a, prefixes), load(args.b, prefixes)

    stats = compare(a, b)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(f"A = {args.a}\nB = {args.b}\n")
        print(stats.to_string(index=False))
        shared = a.index.intersection(b.index)
        for c in stats["column"]:
            if c == "adult_pop" or args.top <= 0:
                continue
            d = (a.loc[shared, c] - b.loc[shared, c]).rename("a_minus_b")
            top = d.abs().sort_values(ascending=False).head(args.top).index
            print(f"\nbiggest movers in {c}:")
            print(pd.DataFrame({"a": a.loc[top, c], "b": b.loc[top, c], "a_minus_b": d[top]}).to_string())
    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    print(f"\ngeoids only in A: {len(only_a)}  {list(only_a[:10])}")
    print(f"geoids only in B: {len(only_b)}  {list(only_b[:10])}")


if __name__ == "__main__":
    main()
