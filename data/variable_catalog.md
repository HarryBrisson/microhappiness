# Variable catalog — every ACS construct + PLACES measure considered

Full breadth, nothing hidden. A direct ACS/PLACES↔happiness correlation isn't computable (GSS
happiness has no tract geography to join on), so for each construct we show its **GSS-fitted
happiness association** where a GSS individual analog exists (held-out single-predictor McFadden
pseudo-R², 1972–2022), and mark the rest **not directly estimable**. `✓ IN` = used in the model;
`○ candidate` = has signal, not yet added; `·` = screened but weak; `✗` = no analog.

## ACS

| variable | construct | GNH domain | GSS analog | happiness assoc (pseudo-R²) | n | status |
|---|---|---|---|---|---|---|
| `B12001` | Marital status | social connectedness | marital | 0.0355 | 75633 | ✓ IN model |
| `B19001` | Household income | material wellbeing | income | 0.0156 | 67843 | ✓ IN model |
| `B11001` | Household type / lives alone | social connectedness | lives_alone | 0.0120 | 72096 | ✓ IN model |
| `B25003` | Home ownership (tenure) | material wellbeing | home_owner | 0.0093 | 37258 | ✓ IN model |
| `B03002` | Race / ethnicity | community vitality | race_ethnicity | 0.0063 | 75699 | ⊘ immutable identity → excluded by policy |
| `B23025` | Employment status | material / time | employment | 0.0034 | 75652 | ✓ IN model |
| `C24010` | Occupation (prestige proxy) | material / time | occ_prestige | 0.0034 | 70360 | · weak alone (dropped) |
| `B11003` | Presence of children | material / social | num_children | 0.0012 | 75407 | · weak alone (dropped) |
| `B01001` | Age | health / psychological | age | 0.0010 | 74829 | ⊘ immutable identity → excluded by policy |
| `B24080` | Self-employment (class of worker) | material wellbeing | self_employed | 0.0007 | 71534 | · weak alone (dropped) |
| `B15003` | Educational attainment | lifelong learning | education | 0.0006 | 75413 | ✓ IN model |
| `B21001` | Veteran status | — | veteran | 0.0001 | 5741 | ⊘ immutable identity → excluded by policy |
| `B01001` | Sex | — | sex | 0.0000 | 75568 | ⊘ immutable identity → excluded by policy |
| `B23022` | Hours worked | time balance | hours_worked | 0.0000 | 43328 | · weak alone (dropped) |
| `B05002` | Nativity (foreign-born) | community vitality | us_born | -0.0000 | 66324 | ⊘ immutable identity → excluded by policy |
| `C17002` | Poverty ratio | material wellbeing | — | — | — | ✗ no GSS analog → not directly estimable |
| `C24030` | Industry | material wellbeing | — | — | — | ✗ no GSS analog → not directly estimable |
| `B14001` | School enrollment | lifelong learning | — | — | — | ✗ no GSS analog → not directly estimable |
| `B05001` | Citizenship | good governance | — | — | — | ✗ no GSS analog → not directly estimable |
| `B16001` | Language at home | culture | — | — | — | ✗ no GSS analog → not directly estimable |
| `B08303` | Commute time | time balance | — | — | — | ✗ no GSS analog → not directly estimable |
| `B08301` | Means of transportation | time balance | — | — | — | ✗ no GSS analog → not directly estimable |
| `B25044` | Vehicles available | material / time | — | — | — | ✗ no GSS analog → not directly estimable |
| `B27001` | Health insurance coverage | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `B25070` | Rent burden | material wellbeing | — | — | — | ✗ no GSS analog → not directly estimable |
| `B25077` | Home value | material wellbeing | — | — | — | ✗ no GSS analog → not directly estimable |
| `B25035` | Housing age (year built) | physical environment | — | — | — | ✗ no GSS analog → not directly estimable |
| `B07003` | Geographic mobility | social connectedness | — | — | — | ✗ no GSS analog → not directly estimable |

## PLACES

| variable | construct | GNH domain | GSS analog | happiness assoc (pseudo-R²) | n | status |
|---|---|---|---|---|---|---|
| `GHLTH` | Fair/poor general health | health | health | 0.0195 | 58448 | ✓ IN model |
| `MHLTH` | Frequent mental distress | health / psychological | mental_health | 0.0185 | 13242 | ✓ IN model |
| `CSMOKING` | Current smoking | health | smoker | 0.0049 | 16373 | ✓ IN model |
| `PHLTH` | Frequent physical distress | health | poor_phys_health | 0.0029 | 9221 | · weak alone (dropped) |
| `BINGE` | Binge drinking | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `LPA` | No leisure physical activity | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `OBESITY` | Obesity | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `SLEEP` | Short sleep (<7h) | health / time | — | — | — | ✗ no GSS analog → not directly estimable |
| `DEPRESSION` | Depression | health / psychological | — | — | — | ✗ no GSS analog → not directly estimable |
| `DIABETES` | Diabetes | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `BPHIGH` | High blood pressure | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `CHD` | Coronary heart disease | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `STROKE` | Stroke | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `CANCER` | Cancer (non-skin) | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `COPD` | COPD | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `CASTHMA` | Current asthma | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `ARTHRITIS` | Arthritis | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `KIDNEY` | Chronic kidney disease | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `TEETHLOST` | All adult teeth lost | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `ACCESS2` | No health insurance | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `CHECKUP` | Routine checkup | health | — | — | — | ✗ no GSS analog → not directly estimable |
| `ISOLATION` | Social isolation (SDOH) | social connectedness | — | — | — | ✗ no GSS analog → not directly estimable |
| `EMOTIONSPT` | Lack of emotional support (SDOH) | social connectedness | — | — | — | ✗ no GSS analog → not directly estimable |
| `FOODINSECU` | Food insecurity (SDOH) | material wellbeing | — | — | — | ✗ no GSS analog → not directly estimable |
| `HOUSINSECU` | Housing insecurity (SDOH) | material wellbeing | — | — | — | ✗ no GSS analog → not directly estimable |
| `DISABILITY` | Any disability | health | — | — | — | ✗ no GSS analog → not directly estimable |
