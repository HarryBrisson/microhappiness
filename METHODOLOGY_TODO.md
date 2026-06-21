# Methodology issues to revisit

Known simplifications and open questions in the current v1 nationwide build. None are blockers — the
estimates validate sensibly against known geography — but each is a place where the model trades rigor
for tractability, and where a v2 could do better. Grouped by priority.

## High priority (affect the numbers people see)

1. **National calibration.** M5's pop-weighted national `pct_very_happy` is ~26.6% vs the raw GSS
   ~29.6% — the negative mental/smoking margins pull the absolute level down ~3 pts. The *relative*
   geographic pattern (what a map shows) is unaffected and validated, but the absolute level is off.
   Fix: a one-parameter national recentering (shift so the pop-weighted mean matches the GSS national
   rate for the vintage year), applied to both index and pct. Cheap; do before headline numbers ship.

2. **No uncertainty intervals.** Estimates are point values. PLACES draws ~1,000 Monte-Carlo datasets
   for CIs; we don't yet. GSS's smaller sample + the graft/rake approximations mean real intervals are
   wide. Add Monte-Carlo over (model coefficients × PLACES margin uncertainty) and publish a CI +
   a minimum credible reporting geography. Until then, treat tract values as indicative, not precise.

3. **Household-vs-person unit mismatch.** Marital/employment are person-level; tenure, income, and
   household-type are household-level; health is adult-level. v1 treats every margin as a proportion
   of the adult population — an approximation (household size correlates with tenure/income, so e.g.
   owner-share over-weights larger households). Fix: convert household margins to person-level using
   ACS household-size-by-type tables, or poststratify at the household level for HH attributes.

## Medium priority (method approximations)

4. **Income harmonization.** GSS income is binned by within-year *percentile* cut points
   (`INCOME_PCT_CUTS`) chosen to approximate the national ACS income-bracket shares; ACS is binned by
   the actual $ brackets. This assumes GSS income rank ≈ ACS income rank and that the percentile cuts
   match the bracket boundaries. Fix: compute the cut points from the ACS national B19001 distribution
   for the vintage year, and convert GSS `coninc` to current dollars rather than ranking.

5. **M5 coefficient grafting.** Mental-health and smoking live in disjoint GSS year-modules (no
   complete cases with the M4 set), so each coefficient is fit on its own co-observed sample and
   *added* to the M4 linear predictor. This assumes additivity (the extra effects don't shift the M4
   coefficients) and that the seed extends by **conditional independence given health**
   (mental ⊥ smoking | health). Fix: proper multiple imputation (MICE) across the modules, or a joint
   model on a harmonized subset; compare against the graft to bound the approximation error.

6. **Seed coverage.** Cells absent from the GSS joint stay at zero weight (we can't rake up mass for
   predictor combinations GSS never observed). Rare combos in unusual tracts are therefore dropped.
   Quantify how much population this loses per tract; smooth the seed (e.g. a log-linear model) if
   material.

7. **Single ordinal outcome modeled as two fits.** We fit a binary very-happy logit *and* a separate
   0/50/100 OLS for the index, rather than one proportional-odds model. They can disagree at the edges.
   Fix: a single ordinal (proportional-odds) model producing both, or check the two are consistent.

8. **PLACES crude vs age-adjusted.** We read the PLACES `data_value` (crude prevalence). Confirm crude
   vs age-adjusted is the right choice and is consistent across GHLTH/MHLTH/CSMOKING; age-adjusted
   would partly re-introduce the age structure we deliberately exclude.

## Lower priority (scope + provenance)

0. **PLACES release coverage gap (KY + PA).** The current CDC PLACES tract release (2025, `cwsq-ngmh`,
   2020 tracts) silently OMITS Kentucky and Pennsylvania — they blanked on the national map. We backfill
   those two states from the 2023 release (`em5e-5hvn`), which uses **2010** tracts, so a fraction of
   their 2020 ACS tracts won't find a health margin and drop out (KY/PA run at lower coverage + a
   slightly older health vintage than the other 48 states). Revisit when CDC restores KY/PA to the
   current release, or add a 2010↔2020 tract crosswalk for the backfill.


9. **Identity-exclusion confound (intentional, document the interpretation).** Age/sex/race are
   excluded as predictors by policy. Their correlation with happiness still flows through the
   *circumstantial* mediators (income, health, marital, employment, ownership) that remain. This is
   the desired behavior — areas are estimated by changeable circumstance, never identity — but it means
   the estimate is "happiness given circumstance," and circumstance is itself unequally distributed.
   Keep this framing explicit in any public writeup.

10. **Temporal panel not built.** v1 uses a single ACS vintage (2022) with coefficients pooled over
    1972–2022. The planned tract×year panel (era-appropriate coefficients × matching ACS vintage,
    period effects, the 2021 GSS mode-break handling, the 2010↔2020 tract crosswalk) is future work.
    NOTE from the trend check: the cross-sectional model's year-to-year correlation is genuinely weak
    (r≈0.11 excluding the 2021/2024 web-mode waves; the 0.36 "all years" figure is inflated by that
    artifact). The 2021 happiness "collapse" is mostly the GSS push-to-web mode change, and `mntlhlth`
    isn't even collected in 2021/2024 — so the trend can't (and shouldn't) be chased with predictors;
    it needs the panel. Unemployment did spike in 2021 (observed) but explains ~1 of the ~12 apparent pts.

11. **Education dropped; employment recode.** Education was screened out as weak (~0) and isn't in the
    v1 cells; revisit if a refined education spec earns its place. Employment is 3-level
    (employed/unemployed/nilf) with armed forces folded into employed — fine, but documented.

12. **Validation — partial.** DONE: national level lands exactly on the GSS rate post-calibration
    (31.0% / index 58.9); convergent validity vs CDC PLACES diagnosed depression (reserved from the
    model) is r≈−0.28 across 78.6k tracts (right direction, modest, as expected). STILL PENDING (gold
    standard): (a) modeled national trend vs the actual GSS `HAPPY` trend by year — needs the temporal
    panel; (b) county aggregates vs the Sharecare/Gallup Community Well-Being Index — needs that licensed
    data; (c) tract life expectancy (USALEEP) — the public SODA copy (5h56-n989) lacks a clean tract
    GEOID (tract number + state/county NAMES only), so it needs the original flat file or a name→FIPS map.

14. **CI scope is partial.** The published 90% CI captures fitted-coefficient sampling uncertainty only
    (~1.5 pt median — tight). It EXCLUDES the PLACES-margin (synthetic-on-synthetic) and structural
    uncertainty, which are larger. Widening to total uncertainty means re-raking per Monte-Carlo draw
    (expensive at 78k tracts) or an analytic margin-sensitivity term — a v2.5 refinement.

13. **GSS weights across years.** We use `wtssps` (falling back to `wtssall`); confirm the weight
    choice is consistent across the pooled decades and post-2018 redesign.
