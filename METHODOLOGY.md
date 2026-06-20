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
| **Age** | U-shaped (dip in midlife) | ✅ | B01001 |
| **Race / ethnicity** | next-tier unconditional gap | ✅ | B03002 |
| **Sex** | small | ✅ | B01001 |
| Disability | negative | ✅ (proxy) | C18120 / B18101 |
| Self-rated health | strong | ❌ | — |
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

## 2. Candidate models (compare, don't guess)

Defined in [`microhappiness/models.py`](microhappiness/models.py). All are deliberately small —
comprehensive ACS *data*, lean *model*.

- **M0 — ceiling probe.** Fullest ACS-derivable set (marital + income + employment + education +
  age + age² + sex + race/ethnicity). Purpose: measure the variance ceiling, not necessarily ship.
- **M1 — minimal / transparent.** `HAPPY ~ marital + income + age + age²`. Two strongest signals +
  the age U-shape. Most interpretable.
- **M2 — PLACES-analog.** Compositional cells `age + sex + race/ethnicity + education` poststratified
  on census joint counts, **plus area covariates** (tract % married, tract median income) and a
  region random effect — mirrors CDC PLACES exactly. Sidesteps joint-table synthesis for the
  hard-to-cross variables by treating them as context.
- **M3 — disciplined-rich / full MRP.** `marital + income + employment + education + age + age² +
  sex + race/ethnicity` as individual effects with **raked** per-area joint poststrat tables +
  region random effects. The fullest defensible individual model.

**Nonlinearities:** age as a quadratic (U-shape); income as decile/log (diminishing returns).
**Honesty:** report McFadden R², calibration, and bootstrap/Monte-Carlo CIs per area; never publish a
point estimate without its interval.

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
   demographic distribution per tract is **synthesized by iterative proportional fitting (raking)**:
   PUMS as the disaggregate seed, ACS published marginals (the tables above) as multi-level controls.
   (PopulationSim / RTI SynthPop are production prior art.) For M2 we skip full synthesis — use the
   census joint cells age×sex×race×education directly + area covariates, PLACES-style.
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
