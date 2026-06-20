# Diagnostics — first real run (GSS cumulative 1972–2022)

Run on the full GSS cumulative file (597MB .dta, N≈75k with HAPPY). These numbers are the empirical
basis for the model design; re-run after any recode change.

> **⊘ EQUITY UPDATE — models are now identity-free.** All production specs exclude immutable identity
> (age, sex, race/ethnicity, nativity, veteran); see METHODOLOGY §1·equity. Dropping identity costs
> only **+0.0041** pseudo-R². Updated identity-free headline numbers: **Step-0 circumstantial ceiling
> = 0.035**; **M1 (marital+income) = 0.038**; **M3 circumstantial-rich = 0.041**; **M4 +health = 0.057**
> (health lift +0.016); mental-health +0.013; smoking +0.004. The identity-inclusive tables below are
> kept for reference (the relationships are unchanged; magnitudes shift ~0.004). The per-variable
> screen still lists identity variables, now marked `⊘ excluded by policy` in `variable_catalog.md`.

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

## Step 2 — candidate-model comparison (pooled 5-fold pseudo-R²)

Each model on its own listwise sample:

| model | predictors | n | pseudo-R² |
|---|---|---|---|
| M1 minimal (marital+income+age) | 3 | 67,374 | 0.042 |
| M0/M3 demographics (rich) | 9 | 30,990 | 0.045 |
| M4 + general health | 10 | 20,003 | **0.060** |
| M5 + health+mental+smoking | 12 | **0 — uncomputable (see below)** | — |

PLACES-margin lift over a co-observed demographic core (each on its own sample, core re-fit on the
same rows for an apples-to-apples delta):

| + margin | n | core R² | +margin R² | lift |
|---|---|---|---|---|
| **health** | 51,879 | 0.051 | **0.066** | **+0.015** |
| **mental_health** | 12,069 | 0.044 | **0.057** | **+0.014** |
| smoking | 14,906 | 0.039 | 0.041 | +0.002 |
| all three together | **0** | — | — | modules don't overlap |

**Reads:**
- **General health is the single biggest lever** (+0.015, ~30% relative over demographics) — the PLACES
  strategy is vindicated. Mental health adds a similar lift on its smaller sample.
- **Smoking adds only +0.002 incrementally** (its univariate 0.005 mostly overlaps demographics/SES).
  Kept as a distinct behavioral marker for *level* estimation, but a minor contributor — set expectations.
- M1 (3 vars) ≈ M0 (9 vars): most demographic signal is in marital+income+age; the extras are thin.
- **KEY CONSTRAINT:** the PLACES predictors are asked in non-overlapping GSS year-modules, so the full
  model has **zero complete cases**. The production fit therefore CANNOT be listwise — it needs multiple
  imputation or per-coefficient partial pooling across modules (a real design requirement for estimate.py).

**Bottom line:** demographics-only ≈ 0.045; + the PLACES health margins → ≈ 0.060–0.066. Still "low"
in absolute terms (happiness is hard), but a meaningful, honest lift. Ship with synthetic-estimate
framing; the health margin is what makes it worth more than a pure-ACS model.
