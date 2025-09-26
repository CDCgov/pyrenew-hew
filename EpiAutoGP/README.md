# EpiAutoGP

Julia-based epidemiological modeling component for the PyRenew-HEW forecasting pipeline using Auto-Gaussian Process methods from the `NowcastAutoGP.jl` package.

## Core Components

- **`run.jl`** - Main entry point orchestrating the complete modeling pipeline
- **`src/`** - Core modeling functionality
  - `input.jl` - JSON data validation and parsing with nowcast structure handling
  - `modelling.jl` - `NowcastAutoGP` model implementation and forecasting
  - `output.jl` - Hubverse-compatible forecast table generation
  - `parse_arguments.jl` - Command-line interface and parameter validation
- **`test/`** - Unit test suite
- **`Containerfile`** - Production container configuration
- **`end-to-end/`** - Complete example demonstrating input data structure, model execution, output structure and visualization from `forecasttools` R package

## Usage

### Command Line Interface

```bash
julia --project=. run.jl \
  --json-input data/input.json \
  --output-dir output/ \
  --n-forecast-weeks 4 \
  --n-particles 4 \
  --n-mcmc 50 \
  --n-hmc 25 \
  --n-forecast-draws 100 \
  --transformation positive \
  --smc-data-proportion 0.2
```

### Key Parameters for entrypoint `run.jl`

- `--json-input`: Path to EpiAutoGPInput-formatted JSON file
- `--output-dir`: Directory for forecast outputs and plots
- `--n-forecast-weeks`: Forecast horizon (default: 4 weeks)
- `--n-particles`: SMC particles for inference (default: 4)
- `--n-mcmc`: MCMC iterations per SMC step (default: 50)
- `--n-hmc`: HMC steps per iteration (default: 25)
- `--n-forecast-draws`: Forecast samples (default: 100)
- `--transformation`: Data transformation (`positive`, `boxcox`, `percentage`)
- `--smc-data-proportion`: Proportion of data per SMC step (default: 0.2)

## JSON Input Format (EpiAutoGPInput)

`EpiAutoGP` requires input data in JSON format that converts directly into the custom `EpiAutoGPInput` struct, along with validation this enforces consistent input. **Converting raw surveillance data to this format is a required preprocessing step** before running the model.

### Required JSON Structure

```json
{
    "data": [7245.0, 7891.0, 8123.0, ...],
    "dates": ["2022-10-01", "2022-10-08", "2022-10-15", ...],
    "disease": "COVID-19",
    "location": "US",
    "forecast_date": "2025-11-08",
    "nowcast_dates": ["2025-11-01"],
    "nowcast": [[7162.9], [7181.2], [7145.8], [7203.1], [7134.5], ...]
}
```

| Field | Type | Description | Requirements |
|-------|------|-------------|--------------|
| `data` | `Vector{Real}` | Time series of surveillance values (e.g., hospital admissions) | Must align with `dates` vector |
| `dates` | `Vector{String}` | ISO date strings for each observation | Weekly intervals, chronologically ordered |
| `disease` | `String` | Disease identifier | Standard disease names (e.g., "COVID-19", "Influenza") |
| `location` | `String` | Geographic location code | FIPS codes or abbreviations (e.g., "US", "CA") |
| `forecast_date` | `String` | Date when forecast is being made | ISO format (YYYY-MM-DD) |
| `nowcast_dates` | `Vector{String}` | Dates for nowcast estimates | Can be empty `[]` if no nowcasts |
| `nowcast` | `Vector{Vector{Real}}` | Nowcast uncertainty samples | Each inner vector = samples for one nowcast date |

### Nowcast Structure Details

The nowcast field uses a nested vector structure where:
- **Inner vector**: One element per nowcast date
- **Outer vector**: Multiple nowcast samples representing uncertainty for the nowcast date(s)
- **Each vector is a nowcast sample**: Contains different realizations of the nowcast value
- **Example**: For 1 nowcast date with 100 samples: `[[sample1], [sample2], ..., [sample100]]`
- **Empty nowcasts**: Use `[]` for nowcast and `[]` for nowcast_dates, this falls back to standard `AutoGP` without nowcast adjustments

### Preprocessing Requirements

1. **Data Alignment**: Ensure `data` and `dates` vectors have the same length
2. **Date Formatting**: Use ISO format (YYYY-MM-DD) for all date fields
3. **Nowcast Generation**: Create uncertainty samples using appropriate statistical models
4. **Validation**: Test JSON structure using `test_json_input.jl` before model runs

## Example Preprocessing Script

The `end-to-end/create_epiautogp_input.py` script demonstrates a possible (simple) preprocessing.

## End-to-End Example

The `end-to-end/` directory contains a complete working example demonstrating the full EpiAutoGP pipeline from raw data to enhanced visualizations.

### End-to-End Example file Structure

```
end-to-end/
├── run_epiautogp_example.sh        # Complete pipeline runner
├── create_epiautogp_input.py       # CSV to JSON conversion with nowcast sampling
├── plot_forecast.R                 # Enhanced multi-layer visualization
├── test_json_input.jl             # JSON format validation
├── vintaged_us_nhsn_data.csv      # Example surveillance data (4102 rows)
└── output/                        # Generated forecasts and plots
    ├── 2025-08-16-CFA-EpiAutoGP-US-covid-nhsn.csv
    └── plots/
        └── forecast_plot_US.png
```

### Running the Example

Run from `/EpiAutoGP`:

```bash
./end-to-end/run_epiautogp_example.sh
```

**NB: When running locally make sure you have generate a local Manifest.toml by running `julia --project=. -e "using Pkg; Pkg.instantiate()"` in the EpiAutoGP directory first. and have R with the `forecasttools` package installed for plotting.**

This pipeline demonstrates:

1. **JSON Conversion**: Transforms data into EpiAutoGPInput format with proper validation
2. **Model Execution**: Runs EpiAutoGP with configurable parameters
3. **Forecast Output**: Generates hubverse-compatible CSV forecasts
4. **Visualization using forecasttools**: Creates multi-layer plots with:
   - **Red line**: Latest available data (extended through forecast period)
   - **Green line**: Vintage data as it appeared on forecast date
   - **Orange point + error bar**: Nowcast uncertainty with IQR
   - **Blue bands**: Forecast predictions with uncertainty bands
   - **Professional legend**: Clear labeling for all data sources

Additionally, you can run the `/EpiAutoGP/end-to-end/create_epiautogp_input.py` script to generate the JSON data and try out other simple nowcasts.

## Unit testing

Test suite with unit tests using Julia's built-in `Test` package run from `/EpiAutoGP` directory:

```bash
# Run all tests
julia --project -e "using Pkg; Pkg.test()"
```

### Test Coverage

- **Input Validation**: JSON structure, data types, nowcast format validation
- **Argument Parsing**: Command-line parameters, defaults, type checking
- **Data Processing**: Transformation methods, edge cases, error handling
- **Output Generation**: Hubverse table format, quantile structures

## Integration Points

- **Input**: EpiAutoGPInput JSON format with vintaged surveillance data
- **Nowcasts**: Vector{Vector{Real}} structure for uncertainty quantification
- **Output**: Hubverse-compatible CSV tables with quantile forecasts
- **Visualization**: R integration via `forecasttools` package

## Container Deployment

EpiAutoGP uses a separate Julia-based container isolated from the main PyRenew-HEW Python/R environment:

```bash
# Build container
docker build -f Containerfile -t epiautogp:latest .

# Run containerized model
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output \
  epiautogp:latest \
  --json-input /data/input.json \
  --output-dir /output
```
