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

| predictor | oos R² | n | era-stable | locally reproducible |
|---|---|---|---|---|
| satfin (financial satisfaction) | 0.052 | 71,013 | ✓ | ❌ not in ACS |
| **marital** | 0.038 | 75,633 | ✓ | ✅ ACS B12001 |
| **health** | 0.020 | 58,448 | ✓ | ✅ PLACES GHLTH |
| **mental_health** | 0.016 | 13,242 | ✓ | ✅ PLACES MHLTH (some years) |
| income | 0.014 | 67,843 | ✓ | ✅ ACS B19001 |
| attend (religion) | 0.012 | 74,966 | ✓ | ❌ |
| race_ethnicity | 0.008 | 75,699 | ✓ | ✅ ACS B03002 |
| polviews (ideology) | 0.003 | 65,878 | ✓ | ❌ |
| age | 0.002 | 74,829 | ✓ | ✅ ACS B01001 |
| education | 0.001 | 75,413 | ✓ | ✅ ACS B15003 |
| sex | 0.000 | 75,568 | ✗ | ✅ ACS B01001 |
| employment | -0.000 | 75,699 | ✗ | ✅ ACS B23025 |

**Reads:**
- The two **PLACES-unlocked predictors (health, mental_health) are the 3rd & 4th strongest overall**
  and the top *non-demographic* locally-reproducible signals — this justifies bringing PLACES in.
- `marital` is the strongest ACS-derivable predictor, confirming the literature.
- The single strongest predictor, `satfin` (financial satisfaction), is **not** ACS-reproducible —
  the gate excludes it, but it hints that subjective financial wellbeing carries more than raw income.
- `age`/`education`/`sex`/`employment` are near-zero *alone* (they may still matter via interactions/
  the U-shape and as poststrat structure) — flagged for the model-comparison stage, not auto-dropped.

**Gated predictor set for the candidate models:** marital, income, race/ethnicity, age (+ sex,
education, employment as poststrat structure) from ACS, **+ health and mental_health from PLACES**.
Next: fit M1–M5 and measure the *combined* ceiling lift from adding the PLACES margins over M0's 0.041.
