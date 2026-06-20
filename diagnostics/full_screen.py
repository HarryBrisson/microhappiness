"""Full transparency catalog: every major ACS construct + PLACES measure vs happiness — hiding nothing.

A *literal* correlation of ACS/PLACES variables with happiness is not computable: GSS happiness has no
tract geography to join onto ACS/PLACES. The honest equivalent, produced here, for every variable
considered:
  - if it has a GSS individual analog -> its ACS-real GSS-happiness association (held-out single-
    predictor McFadden pseudo-R², the same metric as the Step-1 screen);
  - if not -> an explicit "no GSS analog -> not directly estimable".
Every construct is listed with its in/out status, so the breadth (and the exclusions) are fully visible.

Run:  python -m diagnostics.full_screen --gss data/gss_cumulative.dta   ->  data/variable_catalog.md
"""

from __future__ import annotations

from pathlib import Path

from diagnostics.step1_screen import screen
from microhappiness.gss import GSS_COLUMNS, load_gss, recode_predictors

# (system, variable id/table, construct, GNH domain, GSS analog | None)
CATALOG = [
    ("ACS", "B01001", "Age", "health / psychological", "age"),
    ("ACS", "B01001", "Sex", "—", "sex"),
    ("ACS", "B03002", "Race / ethnicity", "community vitality", "race_ethnicity"),
    ("ACS", "B12001", "Marital status", "social connectedness", "marital"),
    ("ACS", "B19001", "Household income", "material wellbeing", "income"),
    ("ACS", "C17002", "Poverty ratio", "material wellbeing", None),
    ("ACS", "B23025", "Employment status", "material / time", "employment"),
    ("ACS", "B24080", "Self-employment (class of worker)", "material wellbeing", "self_employed"),
    ("ACS", "C24010", "Occupation (prestige proxy)", "material / time", "occ_prestige"),
    ("ACS", "C24030", "Industry", "material wellbeing", None),
    ("ACS", "B15003", "Educational attainment", "lifelong learning", "education"),
    ("ACS", "B14001", "School enrollment", "lifelong learning", None),
    ("ACS", "B25003", "Home ownership (tenure)", "material wellbeing", "home_owner"),
    ("ACS", "B11001", "Household type / lives alone", "social connectedness", "lives_alone"),
    ("ACS", "B11003", "Presence of children", "material / social", "num_children"),
    ("ACS", "B23022", "Hours worked", "time balance", "hours_worked"),
    ("ACS", "B05002", "Nativity (foreign-born)", "community vitality", "us_born"),
    ("ACS", "B05001", "Citizenship", "good governance", None),
    ("ACS", "B16001", "Language at home", "culture", None),
    ("ACS", "B21001", "Veteran status", "—", "veteran"),
    ("ACS", "B08303", "Commute time", "time balance", None),
    ("ACS", "B08301", "Means of transportation", "time balance", None),
    ("ACS", "B25044", "Vehicles available", "material / time", None),
    ("ACS", "B27001", "Health insurance coverage", "health", None),
    ("ACS", "B25070", "Rent burden", "material wellbeing", None),
    ("ACS", "B25077", "Home value", "material wellbeing", None),
    ("ACS", "B25035", "Housing age (year built)", "physical environment", None),
    ("ACS", "B07003", "Geographic mobility", "social connectedness", None),
    ("PLACES", "GHLTH", "Fair/poor general health", "health", "health"),
    ("PLACES", "MHLTH", "Frequent mental distress", "health / psychological", "mental_health"),
    ("PLACES", "PHLTH", "Frequent physical distress", "health", "poor_phys_health"),
    ("PLACES", "CSMOKING", "Current smoking", "health", "smoker"),
    ("PLACES", "BINGE", "Binge drinking", "health", None),
    ("PLACES", "LPA", "No leisure physical activity", "health", None),
    ("PLACES", "OBESITY", "Obesity", "health", None),
    ("PLACES", "SLEEP", "Short sleep (<7h)", "health / time", None),
    ("PLACES", "DEPRESSION", "Depression", "health / psychological", None),
    ("PLACES", "DIABETES", "Diabetes", "health", None),
    ("PLACES", "BPHIGH", "High blood pressure", "health", None),
    ("PLACES", "CHD", "Coronary heart disease", "health", None),
    ("PLACES", "STROKE", "Stroke", "health", None),
    ("PLACES", "CANCER", "Cancer (non-skin)", "health", None),
    ("PLACES", "COPD", "COPD", "health", None),
    ("PLACES", "CASTHMA", "Current asthma", "health", None),
    ("PLACES", "ARTHRITIS", "Arthritis", "health", None),
    ("PLACES", "KIDNEY", "Chronic kidney disease", "health", None),
    ("PLACES", "TEETHLOST", "All adult teeth lost", "health", None),
    ("PLACES", "ACCESS2", "No health insurance", "health", None),
    ("PLACES", "CHECKUP", "Routine checkup", "health", None),
    ("PLACES", "ISOLATION", "Social isolation (SDOH)", "social connectedness", None),
    ("PLACES", "EMOTIONSPT", "Lack of emotional support (SDOH)", "social connectedness", None),
    ("PLACES", "FOODINSECU", "Food insecurity (SDOH)", "material wellbeing", None),
    ("PLACES", "HOUSINSECU", "Housing insecurity (SDOH)", "material wellbeing", None),
    ("PLACES", "DISABILITY", "Any disability", "health", None),
]

IN_MODEL = {"marital", "income", "lives_alone", "home_owner", "race_ethnicity", "age", "sex",
            "education", "employment", "health", "mental_health", "smoker"}
INCLUDE_THRESHOLD = 0.005  # single-predictor pseudo-R² floor to flag as a non-trivial signal


def _status(analog, res):
    if analog is None:
        return "—", "—", "✗ no GSS analog → not directly estimable"
    r = res.get(analog)
    if not r or r.get("oos_pseudo_r2") is None:
        return analog, str(r.get("n", 0) if r else 0), "✗ analog too sparse to fit"
    r2, n = r["oos_pseudo_r2"], r["n"]
    if analog in IN_MODEL:
        decision = "✓ IN model"
    elif r2 >= INCLUDE_THRESHOLD:
        decision = "○ candidate (screen ✓)"
    else:
        decision = "· weak alone (dropped)"
    return analog, str(n), f"{decision}"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gss", default="data/gss_cumulative.dta")
    ap.add_argument("--out", default="data/variable_catalog.md")
    args = ap.parse_args()

    df = recode_predictors(load_gss(args.gss, columns=GSS_COLUMNS))
    analogs = sorted({e[4] for e in CATALOG if e[4] and e[4] in df.columns})
    res = {r["var"]: r for r in screen(df, analogs)}

    lines = [
        "# Variable catalog — every ACS construct + PLACES measure considered",
        "",
        "Full breadth, nothing hidden. A direct ACS/PLACES↔happiness correlation isn't computable (GSS",
        "happiness has no tract geography to join on), so for each construct we show its **GSS-fitted",
        "happiness association** where a GSS individual analog exists (held-out single-predictor McFadden",
        "pseudo-R², 1972–2022), and mark the rest **not directly estimable**. `✓ IN` = used in the model;",
        "`○ candidate` = has signal, not yet added; `·` = screened but weak; `✗` = no analog.",
        "",
    ]
    for system in ("ACS", "PLACES"):
        rows = [e for e in CATALOG if e[0] == system]
        # sort: estimable (by assoc desc) first, then no-analog
        def key(e):
            r = res.get(e[4]) if e[4] else None
            a = r["oos_pseudo_r2"] if (r and r.get("oos_pseudo_r2") is not None) else None
            return (a is None, -(a or 0))
        rows.sort(key=key)
        lines += [f"## {system}", "",
                  "| variable | construct | GNH domain | GSS analog | happiness assoc (pseudo-R²) | n | status |",
                  "|---|---|---|---|---|---|---|"]
        for sysname, vid, construct, domain, analog in rows:
            a, n, decision = _status(analog, res)
            r = res.get(analog) if analog else None
            assoc = f"{r['oos_pseudo_r2']:.4f}" if (r and r.get("oos_pseudo_r2") is not None) else "—"
            lines.append(f"| `{vid}` | {construct} | {domain} | {a} | {assoc} | {n} | {decision} |")
        lines.append("")
    Path(args.out).write_text("\n".join(lines))
    n_est = sum(1 for e in CATALOG if e[4] and res.get(e[4], {}).get("oos_pseudo_r2") is not None)
    print(f"wrote {args.out}: {len(CATALOG)} constructs ({n_est} with a GSS analog screened, "
          f"{len(CATALOG) - n_est} marked not-directly-estimable)")


if __name__ == "__main__":
    main()
