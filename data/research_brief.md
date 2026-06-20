# Research brief — microhappiness feasibility

Deep-research pass (2026-06-20): 5 search angles → 21 sources → 87 claims → 25 adversarially
verified (3-vote), 21 confirmed / 4 refuted. Distilled below; full design in `../METHODOLOGY.md`.

## Verified findings (confidence: high unless noted)

1. **Marital status — strongest ACS-derivable predictor.** Married OR≈2.54 (95% CI 2.40–2.67); raw
   happy–unhappy gap ~31pts, stable since the 1970s, survives controls (~24pt conditional). ACS B12001.
   *Peltzman WP#331; Maharlouei PMC7304555.*
2. **Income — on par cross-sectionally** (rich-vs-poor gradient strong at any moment; Easterlin is a
   *time-series* caveat, not load-bearing for poststratification). ACS B19001. *Peltzman. (2-1 vote.)*
3. **Employment + education — positive, modest.** Full-time OR≈1.58; each year educ OR≈1.03. ACS
   B23025 / B15003. *Maharlouei.*
4. **The ceiling: the strongest predictors are NOT in ACS.** Personality (neuroticism Δr²≈0.11),
   self-rated health, social trust (~18pt gap), religious attendance, political ideology (~9pt) all
   meaningful, none ACS-derivable. So an ACS-only model captures demographic variance only.
   *Scientific Reports 2023 (directional, COMPAS-W not GSS); Peltzman.*
5. **CDC PLACES is the exact method twin.** Multilevel logistic per measure: individual
   age/sex/race/education from the survey + a county ACS poverty covariate + state/county random
   effects, applied to census block counts, aggregated to tract/ZCTA. *cdc.gov/places/methodology.*
6. **Tract + ZCTA SAE is achievable** — PLACES does exactly this for adults 18+. *Same.*
7. **PUMA→tract joint via IPF/raking** — PUMS seed + ACS marginals; PopulationSim / RTI SynthPop are
   production prior art. *activitysim populationsim; RTI rti_synth_pop.*
8. **Validation ceiling is real.** MRP correlates r≈0.85–0.95 at county/state but r≈0.28–0.69
   sub-city, indicator-dependent; all evidence is *health*, not wellbeing. *Zhang AJE 2015/2014.*
9. **Synthetic-estimate caveat is mandatory.** PLACES: tract values are "akin to synthetic estimates
   … do not represent the actual observed or direct prevalence." Transfers exactly. *PMC7204458.*

## Refuted (did NOT survive verification)

- Specific numeric "demographics-only ceiling" claims (e.g. "<0.2% unique variance", "68.6% of the
  age curve from non-ACS factors") — **4 claims killed (0-3 / 1-2).** ⇒ **Do not quote a pseudo-R²;
  measure it** (`diagnostics/step0_variance_ceiling.py`).

## Open questions (resolve in-build)

1. **Actual McFadden R² of GSS-HAPPY-on-ACS-demographics** — unestablished; Step-0 honesty gate.
2. **Prior art for US subjective-wellbeing SAE** — none verified (UK ONS local wellbeing is
   survey-direct, not modeled). Do a dedicated check before claiming novelty.
3. **Best external benchmark geography/licensing** — Sharecare CWBI (county) vs PLACES mental-health
   (tract proxy); confirm access.
4. **Uncertainty propagation** — PLACES uses Monte-Carlo (~1,000 sims); GSS's smaller sample + coarser
   geo ⇒ wider CIs and a higher minimum credible reporting geography.

## Key sources

- CDC PLACES methodology — https://www.cdc.gov/places/methodology/index.html
- 500 Cities/PLACES SAE (synthetic caveat) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7204458/
- Zhang et al., MRP validation — https://academic.oup.com/aje/article/182/2/127/93984
- Peltzman, GSS happiness tabulations — https://www.chicagobooth.edu/-/media/research/stigler/pdfs/workingpapers/331peltzmansociopoliticalofhappiness.pdf
- Maharlouei et al., GSS 1972–2018 logit — https://pmc.ncbi.nlm.nih.gov/articles/PMC7304555/
- PopulationSim — https://activitysim.github.io/populationsim/
- RTI SynthPop — https://github.com/RTIInternational/rti_synth_pop
- Sharecare CWBI methods — https://wellbeingindex.sharecare.com/research/community-well-being-index-methods/
- ONS local-authority wellbeing — https://www.ons.gov.uk/datasets/wellbeing-local-authority
