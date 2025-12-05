# EpiAutoGP

A Julia package for epidemiological forecasting using Gaussian Process models with automatic kernel discovery and nowcasting capabilities.

## Overview

Uses [NowcastAutoGP](https://github.com/CDCgov/NowcastAutoGP), it provides an interface for forecasting disease surveillance data with uncertainty quantification via the entrypoint script `run.jl`.

## Installation

This package is part of the PyRenew-HEW forecasting pipeline. To use it:

```bash
cd EpiAutoGP
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

### Running from Command Line

```bash
julia --project=. run.jl \
  --json-input data/input.json \
  --output-dir output/ \
  --n-forecast-weeks 4 \
  --n-forecast-draws 2000 \
  --transformation boxcox
```

### Using as a Julia Package

```julia
using EpiAutoGP
using Dates

# Load input data
input_data = read_and_validate_data("path/to/input.json")

# Generate forecasts
results = forecast_with_epiautogp(
    input_data;
    n_forecast_weeks = 4,
    n_forecasts = 2000,
    transformation_name = "boxcox"
)

# Access results
forecast_dates = results.forecast_dates
forecasts = results.forecasts  # Matrix: (dates × samples)
```

## Input Format

Input data should be provided as JSON with the following structure:

```json
{
  "dates": ["2024-01-01", "2024-01-08", "2024-01-15"],
  "reports": [100.0, 120.0, 95.0],
  "pathogen": "COVID-19",
  "location": "CA",
  "target": "nhsn",
  "forecast_date": "2024-01-15",
  "nowcast_dates": ["2024-01-08", "2024-01-15"],
  "nowcast_reports": [[115.0, 120.0], [90.0, 95.0]]
}
```

The nowcast fields can be empty arrays if no nowcasting is needed/available.

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--json-input` | Path to input JSON file | *Required* |
| `--output-dir` | Output directory for results | *Required* |
| `--n-forecast-weeks` | Number of weeks to forecast | 8 |
| `--n-forecast-draws` | Total number of forecast samples | 2000 |
| `--transformation` | Data transformation (`boxcox`, `positive`, `percentage`) | `boxcox` |
| `--n-particles` | Number of SMC particles | 24 |
| `--smc-data-proportion` | Proportion of data per SMC step | 0.1 |
| `--n-mcmc` | MCMC samples for kernel structure | 100 |
| `--n-hmc` | HMC samples for hyperparameters | 50 |

## Output Format

The model generates Hubverse-compatible forecast files with the following structure:

- Quantile forecasts at standard probability levels
- Point forecasts (median)
- Forecast horizon in weeks from reference date
- Location and pathogen metadata

Output file naming: `{forecast_date}-CFA-EpiAutoGP-{location}-{pathogen}-{target}.csv`

### Forecast Dates

Forecasts include the reference date (week 0) plus `n_forecast_weeks` ahead, this reflects that commonly the forecast date will be a nowcast date.

For example, with `n_forecast_weeks=3`:
- Week 0: `forecast_date`
- Week 1: `forecast_date + 1 week`
- Week 2: `forecast_date + 2 weeks`
- Week 3: `forecast_date + 3 weeks`


## Testing

Run the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
