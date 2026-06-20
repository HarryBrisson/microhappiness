# Diagnostics — first real run (GSS cumulative 1972–2022)

Run on the full GSS cumulative file (597MB .dta, N≈75k with HAPPY). These numbers are the empirical
basis for the model design; re-run after any recode change.

## Step 0 — variance ceiling (the honesty gate)

Full ACS-derivable model **M0** (marital + income + employment + education + age + age² + sex +
race/ethnicity), target = very-happy vs rest:

- **McFadden pseudo-R² = 0.041** (N = 63,102)
- base rate very-happy = 0.297
- predicted P(very happy) spread p10–p90 = **0.164 – 0.432** (a 2.6× range)

**Verdict: LOW ceiling, as predicted — but not useless.** Demographics explain little of happiness
*variance*, yet still separate the most- from least-advantaged compositions ~2.6×. Ship with strong
"synthetic estimate" caveats; do not oversell tract precision. This 0.041 is the number the
literature could not pin (4 ceiling claims were refuted in the research pass).

## Step 1 — broad predictor screen (single-predictor out-of-sample pseudo-R²)

Screened the ACS/PLACES-aligned set + four GNH-USA-domain candidates (home ownership, lives-alone,
children, hours worked) + three non-ACS GSS items as a reference.

| predictor | oos R² | n | era-stable | locally reproducible |
|---|---|---|---|---|
| satfin (financial satisfaction) | 0.049 | 71,013 | ✓ | ❌ not in ACS |
| **marital** | 0.038 | 75,633 | ✓ | ✅ ACS B12001 |
| **health** | 0.020 | 58,448 | ✓ | ✅ PLACES GHLTH |
| **mental_health** | 0.016 | 13,242 | ✓ | ✅ PLACES MHLTH (some years) |
| attend (religion) | 0.015 | 74,966 | ✓ | ❌ |
| **income** | 0.014 | 67,843 | ✓ | ✅ ACS B19001 |
| **lives_alone** *(GNH: social)* | 0.011 | 72,096 | ✓ | ✅ ACS B11001 |
| **home_owner** *(GNH: material)* | 0.010 | 37,258 | ✓ | ✅ ACS B25003 |
| **race_ethnicity** | 0.008 | 75,699 | ✓ | ✅ ACS B03002 |
| polviews (ideology) | 0.004 | 65,878 | ✓ | ❌ |
| age | 0.002 | 74,829 | ✓ | ✅ ACS B01001 |
| num_children *(GNH)* | 0.001 | 75,407 | ✓ | ✅ but ~0 → drop |
| education | 0.001 | 75,413 | ✓ | ✅ ACS B15003 |
| sex | 0.000 | 75,568 | ✗ | ✅ ACS B01001 |
| hours_worked *(GNH: time)* | ~0.000 | 43,328 | ✗ | ✅ but ~0 → drop |
| employment | -0.000 | 75,699 | ✗ | ✅ ACS B23025 (recode weak — see note) |

**Reads:**
- The two **PLACES-unlocked predictors (health, mental_health) are the strongest reproducible
  non-demographic signals** — PLACES justified.
- `marital` leads the ACS predictors, confirming the literature.
- **The GNH lens added two keepers: `lives_alone` (0.011) and `home_owner` (0.010)** — both beat
  race/age/education and have clear mechanisms (social connectedness, material wellbeing). `num_children`
  and `hours_worked` were screened and **disconfirmed** (~0, matching the kids≈null literature) → dropped.
- The single strongest predictor overall, `satfin` (financial satisfaction), is **not** ACS-reproducible
  → gated out (but hints subjective financial wellbeing > raw income).
- **`employment` ≈ 0 is likely a recode artifact** (full-time-vs-rest lumps retirees/students/homemakers;
  the real effect is *unemployed*-vs-not). Recode as a TODO before final model selection.

**Gated reproducible predictor set:** marital, income, lives_alone, home_owner, race/ethnicity, age
(+ sex/education/employment as poststrat structure) from ACS, **+ health & mental_health from PLACES**.
Next: fit M1–M5 and measure the *combined* ceiling lift over M0's 0.041.
