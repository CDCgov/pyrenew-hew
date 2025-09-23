# PyRenew-HEW AI Agent Instructions

## Project Overview
PyRenew-HEW is a CDC epidemiological forecasting system for respiratory diseases (COVID-19, Influenza) using **H**ospital admissions, **E**mergency department visits, and **W**astewater surveillance data. The system combines Python modeling (PyRenew library) with R post-processing in a containerized Azure Batch environment.

## Architecture & Components

### Core Model Structure
- **`pyrenew_hew/`**: Core Python modeling package built on PyRenew library
  - `pyrenew_hew_model.py`: Main Bayesian model with JAX/NumPyro implementation
  - `pyrenew_hew_data.py`: Data preprocessing and validation layer
  - Multi-subpopulation latent infection process with AR(1) dynamics
  - Three observation processes: ED visits, hospital admissions, wastewater concentration

### Pipeline Orchestration
- **`pipelines/`**: Production forecasting workflows
  - `azure_command_center.py`: Interactive job submission interface (preferred method)
  - `batch/setup_job.py`: Azure Batch job configuration for multi-location runs
  - `fit_pyrenew_model.py`: Core model fitting with MCMC sampling
  - Model families: `timeseries`, `pyrenew` with letters `e`, `h`, `he`, `hw`, `hew`

### R Integration (`hewr/`)
- R package for post-processing and visualization
- Integrates with `forecasttools`, `epiprocess`, `epipredict` packages
- Outputs hubverse-compatible forecast tables
- **No Python-R interop via reticulate** - components communicate via files

### EpiAutoGP Component (`EpiAutoGP/`)
- Julia-based epidemiological modeling using AutoGP (Gaussian Process) methods
- **Preferred data manipulation**: Use `TidierData.jl` instead of `DataFrames.jl` for consistency
- **Container isolation**: Separate Julia container from main Python/R environment
- **NowcastAutoGP integration**: Built on the `NowcastAutoGP.jl` package from CDCgov
- **Input/Output compatibility**: Maintains JSON input and hubverse output compatibility with PyRenew pipeline

## Development Workflows

### Container-First Development
```bash
# Use uv for Python dependency management (not pip/conda)
uv sync                          # Install/sync dependencies
uv run python script.py         # Run commands in managed environment

# Container builds with multi-stage approach
make container_build             # Build with Docker/Podman
ENGINE=podman make container_build  # Use Podman instead
```

### Model Execution Patterns
```bash
# Interactive job submission (preferred)
uv run python pipelines/azure_command_center.py

# Direct Makefile execution
make run_hew_model TEST=True MODEL_LETTERS=hew FORECAST_DATE=2025-01-01
make run_timeseries ARGS="--locations-include 'NY GA'"
```

### Testing Strategy
- Tests in `tests/` and `pipelines/tests/`
- R tests in `hewr/tests/testthat/`
- Pytest configuration includes both Python locations
- Always test model fitting with `TEST=True` flag before production

## Code Patterns & Conventions

### Model Configuration
- Model variants controlled by data streams: `fit_ed_visits`, `fit_hospital_admissions`, `fit_wastewater`
- Letters indicate data streams: `e`=ED, `h`=hospital, `w`=wastewater
- Memory requirements: `e`/`h`/`he` use `pyrenew-pool`, `hw`/`hew` use `pyrenew-pool-32gb`

### JAX/NumPyro Patterns
```python
# Use numpyro.deterministic for tracked intermediate values
numpyro.deterministic("rt", inf_with_feedback_proc_sample.rt)

# RandomVariable classes follow this pattern:
class MyProcess(RandomVariable):
    def sample(self, **kwargs):
        # Implementation with scope naming
        with scope(prefix=self.name, divider="_"):
            return result
    
    def validate(self):
        pass
```

### Data Handling
- PyrenewHEWData class manages all input data transformation
- JSON serialization for model inputs: `data_for_model_fit.json`
- Polars for data processing (not pandas)
- MMWR epiweeks for hospital admission aggregation

### Azure Integration
- Environment variables: `NSSP_ETL_PATH`, `PYRENEW_HEW_PROD_OUTPUT_PATH`, `NWSS_VINTAGES_PATH`
- Job naming: `{model_family}-{letters}-{environment}_{date}` format
- Output structure: `{date}_forecasts/` subdirectories

## Key Dependencies & Integration Points

### External Data Sources
- NHSN (hospital admissions): API at `data.cdc.gov`
- NSSP (ED visits): `nssp-etl/gold/` directory structure
- NWSS (wastewater): `nwss_vintages/` with date-stamped directories

### Critical Libraries
- **PyRenew**: Core epidemiological modeling framework (Git dependency)
- **JAX/NumPyro**: Bayesian inference backend
- **Polars**: Fast data processing (preferred over pandas)
- **azuretools**: CFA's Azure Batch utilities
- **forecasttools**: CDC forecast standardization

### Cross-Language Communication
- Python → R: Via JSON/Parquet files (no direct interop)
- R processes: hubverse table generation, visualization
- Container includes both Python (uv) and R (tidyverse) environments

## Common Gotchas

1. **Memory Requirements**: Wastewater models (`w` variants) need 32GB pools
2. **Location Exclusions**: Some states lack data (WY=no E data, TN/ND=no W data)
3. **MMWR Weeks**: Hospital admissions must align to Saturday endings
4. **Container Dependencies**: Both Python and R package installations in Containerfile
5. **Environment Variables**: Azure paths required for production runs
6. **Date Handling**: Model time indexing relative to first data date

## File Naming Conventions
- Python: snake_case throughout
- R: snake_case with underscores in package functions
- Containers: `pyrenew-hew` base name with version tags
- Jobs: descriptive IDs with environment and timestamp suffixes