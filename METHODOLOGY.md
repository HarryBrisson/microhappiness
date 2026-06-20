# microhappiness — methodology

Modeled small-area estimates of subjective happiness for the United States, built by fitting the
**GSS general-happiness question** on individual predictors and **poststratifying onto ACS
demographics** for every census tract / ZCTA. This is the same method CDC PLACES uses for health
indicators, with `GSS HAPPY` in place of `BRFSS`.

> **These are synthetic estimates.** Each value is the happiness we'd *expect* given an area's
> demographic composition — **not** an observed local measurement. (CDC PLACES uses this exact
> framing for its tract estimates: "akin to synthetic estimates … they do not represent the actual
> observed or direct prevalence in a census tract.")

## The outcome

GSS `HAPPY` (NORC GSS Data Explorer variable 434): *"Taken all together, how would you say things
are these days — would you say that you are very happy, pretty happy, or not too happy?"* Asked on
(nearly) every GSS wave since 1972 — tens of thousands of pooled responses.

We model it two ways and publish both:
- **`pct_very_happy`** — P(very happy), a binary logit (the simplest, most legible target).
- **`happiness_index`** — E[score] on a 0/50/100 coding (not too / pretty / very), i.e. an ordinal
  (proportional-odds) model collapsed to a 0–100 index.

## 1. Predictors — and the honest ceiling

Robust GSS / wellbeing-literature predictors, split by whether ACS can supply them per area:

| Predictor | Direction / effect | In ACS? | ACS table |
|---|---|---|---|
| **Marital status** | strongest demographic; married OR≈2.54, ~31pt raw gap, stable since 1970s | ✅ | B12001 |
| **Household income** | strong cross-sectional gradient, on par with marital | ✅ | B19001 / B19013 |
| **Employment (full-time)** | OR≈1.58 | ✅ | B23025 |
| **Education** | modest, OR≈1.03/yr | ✅ | B15003 |
| Age | U-shaped (dip in midlife) | ✅ but ⊘ **excluded (identity)** | B01001 |
| Race / ethnicity | next-tier unconditional gap | ✅ but ⊘ **excluded (identity)** | B03002 |
| Sex | small | ✅ but ⊘ **excluded (identity)** | B01001 |
| Nativity / veteran | small | ✅ but ⊘ **excluded (identity)** | B05002 / B21001 |
| Disability | negative | ✅ (proxy) | C18120 / B18101 |
| **Self-rated health** | strong | ✅ **via PLACES** | CDC PLACES `GHLTH`, tract/ZCTA (see §1a) |
| Religious attendance | positive | ❌ | — |
| Social trust | ~18pt gap | ❌ | — |
| Political ideology | conservatives ~9pt happier | ❌ | — |
| Personality (neuroticism…) | **dominant** between-person predictor | ❌ | — |

**The ceiling.** The single most powerful happiness predictors are *not* ACS-derivable. An
ACS-only model therefore captures demographic variance only — it will produce smooth,
composition-driven estimates, not dramatic neighborhood contrast. The exact share of variance the
ACS predictors reach for `HAPPY` **was not reliably established in the literature** (multiple
attempts to pin a number were contradicted). So we **measure it ourselves first** — see
`diagnostics/step0_variance_ceiling.py`. If McFadden's pseudo-R² is near-zero, we reframe the
product (or stop) rather than ship false precision.

### 1·equity. Circumstantial predictors only — no immutable identity

**Policy:** we model happiness from **mutable, circumstantial** factors (income, marital/household
status, employment, education, home ownership, health, mental health) and **never** from **immutable
identity** characteristics (age, sex, race/ethnicity, nativity, veteran status — `IMMUTABLE_IDENTITY`
in `models.py`). Two reasons: (1) a neighborhood can't change its identity composition, so an
identity-driven estimate is unactionable and would "reward"/"penalize" areas for who lives there; (2)
the estimate should track **conditions you could improve**, not demographics.

This is nearly free, and we measured it: dropping every identity predictor costs only **+0.0041**
pseudo-R² (the identity variables are weak — race 0.006, age 0.001, sex 0.000), while the real levers
(marital, income, health) are all circumstantial. If an area's identity composition correlates with
happiness, that signal flows through its **circumstances** (income, health) — so a $50k household is
estimated the same wherever it is, and the map reflects material conditions, not skin color or age.
Identity variables remain in the transparency catalog, clearly marked `⊘ excluded by policy`, so the
exclusion is visible rather than hidden. (Marital status and education are classified as
circumstantial/socioeconomic, not identity — flag if you'd draw either line differently.)

### 1a. PLACES raises the ceiling — by supplying the health *marginal*

The most valuable non-ACS predictor we can recover is **self-rated health**. CDC PLACES publishes
modeled health prevalence at **tract and ZCTA** (our exact geography), including `GHLTH` =
% fair-or-poor self-rated health. The clean way to use it is *not* as a generic feature but as a
**poststratification marginal**:

- GSS carries individual self-rated health (`HEALTH`), so we **fit** `HAPPY ~ … + health`.
- PLACES gives each area's % fair-or-poor health, which we **rake into the per-area joint table** as
  the health margin (ACS supplies the demographic margins). The national **seed** is GSS/PUMS
  microdata, which already carries the demographics×health correlation — so no observed joint is
  needed. **GSS is thus both the fitted model and the joint seed; ACS + PLACES are the margins.**

This lets us both fit on and locally distribute the strongest predictor ACS lacks — something a
pure-ACS model cannot do. Caveats: PLACES is itself modeled (BRFSS→ACS), so this is
**synthetic-on-synthetic** — compound and label the uncertainty; collapse GSS `HEALTH` to PLACES's
"fair-or-poor" so the control total matches.

**Mental health is a distinct axis, not circular.** Wellbeing science separates *life-evaluation*
(what GSS `HAPPY` measures) from *affect*, *eudaimonia*, and *illbeing* — one can carry a depression
diagnosis and still evaluate life as "pretty happy." So PLACES `MHLTH` (poor-mental-health days) is a
**legitimate predictor** (candidate **M5**), fit on GSS's individual `MNTLHLTH` item (asked only in
some years → smaller fitting sample) and raked from the PLACES margin. We deliberately keep **M1–M4
as variants that *exclude* mental health** — both as a cautious baseline and so `MHLTH` can serve as
an *independent* validation target for them (§5). Depression has no clean GSS individual analog to fit
on, so it stays validation-only.

## 2. Candidate models (compare, don't guess)

Defined in [`microhappiness/models.py`](microhappiness/models.py). All are deliberately small —
comprehensive ACS *data*, lean *model*.

All specs are **identity-free** (§1·equity). Measured ceilings (pooled 5-fold pseudo-R²) in brackets.

- **M0 — circumstantial ceiling probe** [0.035]. Fullest circumstantial set (marital + income +
  employment + education + home_owner + lives_alone). Purpose: the step-0 honesty gate.
- **M1 — minimal / transparent** [0.038]. `HAPPY ~ marital + income`. The two strongest circumstantial
  signals; nearly as good as M0 with 2 variables.
- **M2 — area-covariate.** A few individual circumstantial cells (marital + income + employment) **plus
  area-level circumstantial covariates** (tract median income, % owner-occupied, % fair-poor health) +
  region random effect — PLACES-style, sidesteps full joint synthesis. Covariates are all actionable,
  never identity composition.
- **M3 — circumstantial-rich** [0.041]. marital + income + employment + education + home_owner +
  lives_alone, with **raked** per-area joint tables + region random effects.
- **M4 — + self-rated health (PLACES-unlocked)** [0.057]. M3 **+ health**, fit on GSS `HEALTH` and
  raked from the PLACES `GHLTH` marginal (§1a). The single biggest lever (+0.016).
- **M5 — + mental-health + smoking.** M4 **+ mental-health-days** (GSS `MNTLHLTH` ↔ PLACES `MHLTH`,
  +0.013) **+ smoking** (GSS `SMOKE` ↔ PLACES `CSMOKING`, +0.004 — a distinct level marker). M1–M4
  exclude mental health so PLACES `MHLTH` can independently validate them.

**Nonlinearities:** income as decile/log (diminishing returns).
**Honesty:** report McFadden R², calibration, and bootstrap/Monte-Carlo CIs per area; never publish a
point estimate without its interval.

### 2a. Variable selection — broad screen, then a theory gate

We don't trust the literature's predictor shortlist blindly. A **broad empirical screen**
(`diagnostics/step1_screen.py`) runs over GSS's own wide variable set — the only place individual
happiness and a candidate predictor sit together (ACS/PLACES have no fine geography to attach to a
GSS respondent). It ranks candidates by **cross-validated incremental** signal (held-out split, era
stability) so the breadth doesn't chase noise. A variable enters the **final** model only if it is
simultaneously (a) empirically useful, (b) reproducible at tract/ZCTA via ACS or PLACES, and (c)
backed by a **clear, logical mechanism** — a human call that keeps the model parsimonious and guards
against spurious correlations. This re-ranks predictors against the data while still refusing the
kitchen sink.

We use the **GNH-USA domains** (psychological wellbeing, social connectedness, material wellbeing,
health, time balance, religion/spirituality, lifelong learning, good governance, community vitality,
physical environment, culture) as the *breadth checklist* — for each domain, is there a GSS item to
fit on AND a local (ACS/PLACES) source to reproduce? Most fail one side (religion, governance,
environment, culture have no tract-level reproducible source, or no GSS analog), but the screen
confirmed two **material/social** additions that clear both bars and the data: **home ownership**
(GNH material; ACS B25003) and **living alone** (GNH social; ACS B11001). `num_children` and
`hours_worked` were screened and disconfirmed (~0), so dropped. The non-reproducible-but-strong
domains (religion, governance, environment) are candidates for coarse *area-level covariates* later.

## 3. Temporal dimension (decades of GSS × ACS vintages)

GSS runs 1972→present; ACS gives matching demographics per year (5-yr tract from 2009; decennial
before). So we **don't freeze one mapping** — we fit on the pooled series with **year random effects
+ `marital×era` and `income×era` interactions**, then apply the era-appropriate coefficients to the
era-appropriate ACS vintage. Output is a **tract × year panel** (not a single snapshot), which drops
straight into Penlight's existing metric-timeline machinery.

This buys a **second, independent validation axis**: aggregate modeled tracts to the nation per year
and check the synthetic national trend reproduces the *actual* GSS national happiness trend (known
year by year) — orthogonal to the cross-sectional county benchmark.

**Temporal cautions baked in:** the 2021 GSS mode shift (web/COVID) is a documented break — flag /
down-weight it; age–period–cohort effects are under-identified — stay modest; tract boundaries &
PUMA defs change across vintages — a 2010↔2020 tract crosswalk is required for any true time series;
ACS 5-yr "years" are overlapping windows — label them as such.

## 4. Small-area estimation pipeline

1. **Fit** the chosen model on GSS microdata (with survey weights `WTSSALL`/`WTSSPS`).
2. **Poststratification table per area.** PUMS microdata is only identified to PUMA, so the joint
   distribution per tract is **synthesized by iterative proportional fitting (raking)**: a national
   **seed** (GSS, or PUMS) carrying the predictor correlations, raked to each area's published
   marginals — ACS for demographics + **PLACES for the health margin** (§1a). Using GSS as the seed
   keeps `health` in the joint even though ACS/PUMS lack it. (PopulationSim / RTI SynthPop are
   production prior art.) For M2 we skip full synthesis — use the census joint cells
   age×sex×race×education directly + area covariates, PLACES-style.
3. **Predict** each cell's happiness, multiply by the cell's ACS population, sum → area estimate.
4. **Aggregate / allocate** tract estimates up to any geography (ward, community area, χGRID, county).
5. **Uncertainty:** Monte-Carlo simulation over the fitted model (PLACES draws ~1,000 datasets);
   propagate to per-area CIs. GSS's smaller sample → wider intervals than PLACES; set a minimum
   credible reporting geography from the CI widths.

## 5. Validation

- **County cross-section:** correlate county aggregates against the **Sharecare Community Well-Being
  Index** (county/community) and Gallup wellbeing. Expect strong county r, weaker sub-city.
- **Tract proxy:** correlate against **CDC PLACES** poor-mental-health / poor-physical-health days
  (tract/ZCTA) — adjacent, not identical.
- **Longitudinal:** synthetic national trend vs actual GSS national `HAPPY` trend by year.
- Report all three honestly; treat sub-city agreement as the weak point it is.

## 6. Data access

- **GSS:** the cumulative datafile 1972–present (NORC `gss.norc.org`, Stata/SPSS), or the `gssr` R
  package, or the GSS Data Explorer. Pull `HAPPY` + `MARITAL`, `REALINC`/`INCOME`, `WRKSTAT`, `EDUC`,
  `AGE`, `RACE`/`HISPANIC`, `SEX`, `YEAR`, weights.
- **ACS:** Census API (`api.census.gov/data/<year>/acs/acs5`). Poststrat marginals & covariates:
  B12001 (marital), B19001/B19013 (income), B23025 (employment), B01001 (age×sex), B15003
  (education), B03002 (race/ethnicity), C18120/B18101 (disability). Tract and ZCTA both available.
  Census joint cells for M2 come from the decennial / detailed ACS tables. Needs a Census API key.
- **CDC PLACES:** the PLACES API (`data.cdc.gov`), measure `GHLTH` (fair-or-poor self-rated health) at
  tract & ZCTA — the health poststrat margin (§1a). `MHLTH`/`DEPRESSION` pulled for validation only.

## Integration with Penlight (ward-wise)

Consumed as a **published artifact**, not a code dependency (keeps statsmodels + the ACS pulls out of
the Lambda). microhappiness publishes a versioned **nationwide tract table** (`GEOID, year,
happiness_index, pct_very_happy, se, adult_pop`) + a `byop/v1`-style `aggregation_spec.json`.
ward-wise ingests it on the same tract→ward/CA/χGRID path it already uses for ACS/PLACES metrics,
auto-labeled "modeled estimate." Because the model is national, **Rob's Columbus is free** — same
table, Columbus tracts.

## Sources

- CDC PLACES methodology — https://www.cdc.gov/places/methodology/index.html
- 500 Cities / PLACES SAE (synthetic-estimate caveat; validity weaker sub-city) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7204458/
- Zhang et al., MRP validation, county r≈0.85 — https://academic.oup.com/aje/article/182/2/127/93984
- Peltzman, *The Socio-Political Demography of Happiness* (GSS HAPPY tabulations) — https://www.chicagobooth.edu/-/media/research/stigler/pdfs/workingpapers/331peltzmansociopoliticalofhappiness.pdf
- Maharlouei et al., GSS 1972–2018 happiness logit (marital/employment/education ORs) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7304555/
- PopulationSim (IPF/raking population synthesis) — https://activitysim.github.io/populationsim/
- RTI SynthPop (PUMA→tract synthetic population) — https://github.com/RTIInternational/rti_synth_pop
- Sharecare Community Well-Being Index methods — https://wellbeingindex.sharecare.com/research/community-well-being-index-methods/
- ONS local-authority wellbeing — https://www.ons.gov.uk/datasets/wellbeing-local-authority

*Full research brief (verified claims + refutations + caveats) archived in `data/research_brief.md`.*
