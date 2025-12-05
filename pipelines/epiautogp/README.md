# EpiAutoGP Integration Module

This module provides an interface with the `EpiAutoGP` model within the pyrenew-hew pipeline.

## Components

### `prep_epiautogp_data.py`

Contains the main data conversion function:

- **`convert_to_epiautogp_json()`**: Converts surveillance data to EpiAutoGP JSON format
  - Supports both NSSP (ED visit percentages) and NHSN (hospital admission counts)
  - Handles nowcast data when available
  - Validates input parameters and data availability
