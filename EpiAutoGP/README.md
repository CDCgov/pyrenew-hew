# EpiAutoGP

Julia-based epidemiological modeling component for the PyRenew-HEW forecasting pipeline using Auto-Gaussian Process methods.

## Overview

EpiAutoGP provides nowcasting and forecasting capabilities using the `NowcastAutoGP.jl` package. It integrates with the PyRenew-HEW system by accepting JSON input data and producing hubverse-compatible forecast outputs.

## Key Components

- `run.jl` - Main entry point for model execution
- `parse_arguments.jl` - Command-line argument parsing
- `Project.toml` - Julia package dependencies
- `Containerfile` - Docker container configuration

## Usage

```bash
julia --project=. run.jl \
  --json-input data/input.json \
  --output-dir output/ \
  --disease COVID-19 \
  --location CA \
  --forecast-date 2024-12-21
```

## Testing

Simple test suite using Julia's built-in `Test` package:

```bash
# Run tests
julia --project=. test/runtests.jl

# Or using Pkg
julia --project=. -e "using Pkg; Pkg.test()"
```

**Test approach:** Focused on core functionality without complex infrastructure. Tests validate argument parsing, default values, and type checking using mock command-line arguments.

## Data Preference

Uses `TidierData.jl` instead of `DataFrames.jl` for R-like syntax and tidyverse compatibility.

## Container Deployment

Separate Julia-based container isolated from the main PyRenew-HEW Python/R environment for clean dependency management.
