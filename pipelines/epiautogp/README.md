# EpiAutoGP Integration Module

This module provides an interface with the `EpiAutoGP` model within the pyrenew-hew pipeline.

## Components

### `prep_epiautogp_data.py`

Contains the main data conversion function:

- **`convert_to_epiautogp_json()`**: Converts surveillance data to EpiAutoGP JSON format
  - Supports both NSSP (ED visits) and NHSN (hospital admission counts)
  - Supports daily and epiweekly data frequencies
  - Option to convert ED visits to percentages of total ED visits
  - Handles both legacy JSON format and TSV files
  - Validates input parameters and data availability

## Data Sources

The function can read from two types of data sources:

### 1. Legacy JSON Format
- `data_for_model_fit.json` file containing `nssp_training_data` and `nhsn_training_data`

### 2. TSV Files (Recommended)
- **Daily data**: `combined_training_data.tsv`
  - Contains: `observed_ed_visits`, `other_ed_visits`, `observed_hospital_admissions`
- **Epiweekly data**: `epiweekly_combined_training_data.tsv`
  - Contains: `observed_ed_visits`, `other_ed_visits`, `observed_hospital_admissions`


## Output Format

The function generates a JSON file with the following structure:

```json
{
  "dates": ["2024-09-22", "2024-09-23", ...],
  "reports": [45.5, 52.3, ...],
  "pathogen": "COVID-19",
  "location": "DC",
  "target": "nssp",
  "forecast_date": "2024-12-20",
  "nowcast_dates": [],
  "nowcast_reports": []
}
```
