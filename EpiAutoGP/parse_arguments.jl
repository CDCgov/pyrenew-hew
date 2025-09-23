using ArgParse
using Dates

"""
    parse_arguments()

Parses command-line arguments for the EpiAutoGP model.

# Arguments

- `--json-input::String` (required): Path to JSON file containing model input data.
- `--output-dir::String` (required): Directory for saving model outputs.
- `--forecast-date::Date` (required): Reference date for forecasting in `YYYY-MM-DD` format.
- `--n-forecast-weeks::Int` (default: 4): Number of weeks to forecast.
- `--n-particles::Int` (default: 24): Number of particles for SMC.
- `--n-mcmc::Int` (default: 100): Number of MCMC steps for GP kernel structure.
- `--n-hmc::Int` (default: 50): Number of HMC steps for GP kernel hyperparameters.
- `--n-forecast-draws::Int` (default: 2000): Number of forecast draws.
- `--n-redact::Int` (default: 1): Number of weeks to redact for nowcasting.
- `--transformation::String` (default: "boxcox"): Data transformation type ("boxcox", "positive", "percentage").

# Returns

A dictionary containing the parsed command-line arguments.
"""
function parse_arguments()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--json-input"
            help = "Path to JSON file containing model input data"
            arg_type = String
            required = true
        "--output-dir"
            help = "Directory for saving model outputs"
            arg_type = String
            required = true
        "--forecast-date"
            help = "Reference date for forecasting (YYYY-MM-DD format)"
            arg_type = Date
            required = true
        "--n-forecast-weeks"
            help = "Number of weeks to forecast"
            arg_type = Int
            default = 4
        "--n-particles"
            help = "Number of particles for SMC"
            arg_type = Int
            default = 24
        "--n-mcmc"
            help = "Number of MCMC steps for GP kernel structure"
            arg_type = Int
            default = 100
        "--n-hmc"
            help = "Number of HMC steps for GP kernel hyperparameters"
            arg_type = Int
            default = 50
        "--n-forecast-draws"
            help = "Number of forecast draws"
            arg_type = Int
            default = 2000
        "--n-redact"
            help = "Number of weeks to redact for nowcasting"
            arg_type = Int
            default = 1
        "--transformation"
            help = "Data transformation type (boxcox, positive, percentage)"
            arg_type = String
            default = "boxcox"
    end

    return parse_args(s)
end
