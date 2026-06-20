# microhappiness 🙂

Modeled small-area estimates of **subjective happiness** for the United States — a happiness number
for every census tract / ZCTA (and, by aggregation, any ward, neighborhood, or county), built from
the **GSS happiness question** poststratified onto **ACS** demographics.

It's the [CDC PLACES](https://www.cdc.gov/places/methodology/index.html) method (multilevel
regression + poststratification) pointed at wellbeing instead of health. National by construction, so
the same published table works for any US city.

> ⚠️ **Synthetic estimates, not measurements.** Each value is the happiness *expected* given an
> area's demographic mix — not an observed local survey. Most of what actually drives happiness
> (personality, health, social trust, faith) isn't in the ACS, so these estimates are smooth and
> composition-driven by design. We measure that ceiling honestly before trusting any output
> (`diagnostics/step0_variance_ceiling.py`). Read [METHODOLOGY.md](METHODOLOGY.md) first.

## Status

Early scaffold. The design is fixed (see METHODOLOGY.md); implementation is staged behind a Step-0
honesty gate.

## How it works (4 steps)

1. **Fit** a small, transparent model of GSS `HAPPY` on ACS-derivable predictors (marital, income,
   employment, education, age, race/ethnicity). Several [candidate specs](microhappiness/models.py)
   — minimal → PLACES-analog → disciplined-rich — are compared, not guessed.
2. **Poststratify** onto each area's ACS demographic distribution (synthesizing the per-tract joint
   table by raking where needed).
3. **Predict + aggregate** to a nationwide tract/ZCTA table, then up to any polygons.
4. **Validate** against Sharecare/Gallup county wellbeing + CDC PLACES mental-health, and against the
   GSS national trend over time.

Decades of GSS × yearly ACS make this a **tract × year panel**, not a snapshot — see the temporal
section in METHODOLOGY.md.

## Quickstart (planned)

```bash
pip install -e .
export CENSUS_API_KEY=...                       # https://api.census.gov/data/key_signup.html
python -m diagnostics.step0_variance_ceiling     # measure the ceiling FIRST (go/no-go)
python -m microhappiness.run --model m1_minimal --acs-year 2022 --geo tract
```

## Layout

```
microhappiness/
  models.py        candidate model specs (compare these)
  gss.py           fetch + recode GSS microdata (HAPPY + predictors)
  acs.py           Census API: poststrat marginals + area covariates (tract/ZCTA)
  poststratify.py  IPF / raking -> per-area joint demographic tables
  estimate.py      fit model, predict cells, aggregate to areas + uncertainty
  validate.py      benchmark vs Sharecare/Gallup + CDC PLACES + GSS national trend
  publish.py       nationwide tract table + byop/v1 aggregation_spec.json
diagnostics/
  step0_variance_ceiling.py   measure McFadden R² of ACS-only model (honesty gate)
```

## For collaborators (e.g. Columbus)

Nothing here is Chicago-specific. The published table is nationwide tracts — point it at your city's
tract geometries and aggregate. Consumed as a pinned published artifact, so you don't import this
repo's modeling stack.

## License / data

Public. GSS data © NORC (cumulative datafile, public use). ACS via the Census API. Estimates are
modeled; cite METHODOLOGY.md and the synthetic-estimate caveat in any downstream use.
