# Comprehensive Variable Naming Guide for PyRenew HEW

## Naming Conventions

### 1. Type-First Ordering
**Pattern:** `{data_type}_{property}`

All variables lead with the data type (`ed`, `hosp_admit`, `ww`) followed by the property.

### 2. Time Index Variables
**Pattern:** `t_{description}`

Variables storing integer time indices use `t_` prefix (not `model_t_`).

### 3. Standard Abbreviations
- `ww` = wastewater
- `obs` = observation/observed
- `hosp_admit` = hospital admission
- `ed` = emergency department
- `lod` = limit of detection
- `conc` = concentration
- `subpop` = subpopulation (singular)
- `t` = time index in model coordinate system

### 4. Omit Redundant Qualifiers
- Remove "disease" (always COVID-related)
- Remove "observed" from data class properties (implicit)
- Remove "model" from time indices (`model_t` → `t_`)

## Complete Mapping by File

### pyrenew_hew_data.py

#### Date Properties

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `dates_observed_ed_visits` | `ed_dates` | `{type}_dates` |
| `dates_observed_hospital_admissions` | `hosp_admit_dates` | `{type}_dates` |
| `dates_observed_disease_wastewater` | `ww_dates` | `{type}_dates` |
| `first_ed_visits_date` | `ed_first_date` | `{type}_first_date` |
| `last_ed_visits_date` | `ed_last_date` | `{type}_last_date` |
| `first_hospital_admissions_date` | `hosp_admit_first_date` | `{type}_first_date` |
| `last_hospital_admissions_date` | `hosp_admit_last_date` | `{type}_last_date` |
| `first_wastewater_date` | `ww_first_date` | `{type}_first_date` |
| `last_wastewater_date` | `ww_last_date` | `{type}_last_date` |

#### Observation Data

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `data_observed_disease_ed_visits` | `ed_visit_count` | `{type}_count` |
| `data_observed_disease_hospital_admissions` | `hosp_admit_count` | `{type}_count` |
| `data_observed_total_ed_visits` | `total_ed_visit_count` | `total_{type}_count` |
| `data_observed_disease_wastewater_conc` | `log_ww_conc_obs` | `{transform}_{type}_{property}` |

#### Time Index Properties

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `model_t_obs_wastewater` | `ww_obs_time` | `{type}_obs_time` |
| `model_t_obs_ed_visits` | `ed_obs_time` | `{type}_obs_time` |
| `model_t_obs_hospital_admissions` | `hosp_admit_time` | `{type}_time` |

#### Observation Units/Indices

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `ww_observed_subpops` | `ww_obs_subpop` | `{type}_obs_{property}` (singular) |
| `ww_observed_lab_sites` | `ww_obs_unit` | `{type}_obs_{property}` |
| `ww_log_lod` | `log_lod` | `{transform}_lod` |

### pyrenew_hew_model.py

#### WastewaterObservationProcess.sample() - Local Variables

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `which_obs_t_viral_genome` | `expected_conc_time_idx` | `{description}_idx` |
| `expected_obs_viral_genomes` | `expected_log_conc` | Remove "obs" from predictions |
| `model_t_first_latent_viral_genome` | `t_first_viral_genome` | `t_{description}` |

#### WastewaterObservationProcess.sample() - Parameters

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `ww_model_t_observed` | `ww_obs_time` | Match data class property |
| `ww_observed_subpops` | `ww_obs_subpop` | Match data class property |
| `ww_observed_lab_sites` | `ww_obs_unit` | Match data class property |
| `ww_log_lod` | `log_lod` | Match data class property |
| `model_t_first_latent_infection` | `t_first_infection` | `t_{description}` |

#### EDVisitObservationProcess.sample() - Parameters

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `model_t_observed` | `ed_obs_time` | Match data class property |
| `model_t_first_latent_infection` | `t_first_infection` | `t_{description}` |

#### HospAdmitObservationProcess.sample() - Parameters

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `model_t_observed` | `hosp_admit_time` | Match data class property |
| `model_t_first_latent_infection` | `t_first_infection` | `t_{description}` |

#### HospAdmitObservationProcess.calculate_weekly_hosp_indices() - Parameters

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `model_t_first_latent_admissions` | `t_first_admissions` | `t_{description}` |
| `model_t_observed` | `hosp_admit_time` | Match data class property |

#### PyrenewHEWModel.sample() - Data Access

| Old Name | New Name | Pattern |
|----------|----------|---------|
| `data.data_observed_disease_ed_visits` | `data.ed_visit_count` | Match property name |
| `data.data_observed_disease_hospital_admissions` | `data.hosp_admit_count` | Match property name |
| `data.model_t_obs_ed_visits` | `data.ed_obs_time` | Match property name |
| `data.model_t_obs_hospital_admissions` | `data.hosp_admit_time` | Match property name |
| `data.model_t_obs_wastewater` | `data.ww_obs_time` | Match property name |
| `data.ww_observed_lab_sites` | `data.ww_obs_unit` | Match property name |
| `data.ww_log_lod` | `data.log_lod` | Match property name |

## Summary by Change Type

### Round 1: Original Refactoring (from Stan alignment)
- Hospital admissions: `oht` → `n_hosp_admit_obs`, `hosp_times` → `hosp_admit_time`, `hosp` → `hosp_admit_count`
- Wastewater: `owt` → `n_ww_obs`, various `ww_sampled_*` → `ww_obs_*`, `log_conc` → `log_ww_conc_obs`
- Predictions: clarify observations vs predictions

### Round 2: Consistency & Concision
- Remove "disease" qualifier everywhere
- Standardize to type-first ordering for dates
- Use "obs" not "observed"
- Singular "subpop" not plural "subpops"

### Round 3: Time Index Clarity
- Replace `model_t_` prefix with `t_` prefix
- Method parameters match data class property names
- Reserve `_idx` suffix for computed indexing variables

## Pattern Examples

### Good Patterns
```python
# Data properties
ww_obs_time          # wastewater observation time indices
ed_visit_count       # emergency department visit counts
hosp_admit_dates     # hospital admission calendar dates

# Time indices
t_first_infection    # time index of first infection
t_first_viral_genome # time index when genome shedding starts

# Computed indices
expected_conc_time_idx = ww_obs_time - t_first_viral_genome

# Method calls
obs_process.sample(
    ww_obs_time=data.ww_obs_time,
    t_first_infection=-50,
    ...
)
```

### Patterns to Avoid
```python
# Bad: property-first
dates_observed_ed_visits          # should be: ed_dates

# Bad: redundant qualifiers
data_observed_disease_ed_visits   # should be: ed_visit_count
model_t_first_latent_infection    # should be: t_first_infection

# Bad: inconsistent plural/singular
ww_observed_subpops               # should be: ww_obs_subpop

# Bad: ambiguous prefixes
model_t_observed                  # should be: ww_obs_time (or ed_obs_time)
```

## Variables Not Changed

These retain their current names:
- `n_subpops`, `n_ww_lab_sites` - count variables already clear
- `lab_site_to_subpop_map` - mapping clearly named
- `ww_censored`, `ww_uncensored` - censoring arrays clear
- `censored_idx`, `uncensored_idx` - index arrays clear
- `population_size` - standard term
- `first_data_date_overall`, `last_data_date_overall` - aggregate properties clear
