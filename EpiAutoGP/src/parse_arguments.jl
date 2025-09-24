"""
    parse_arguments()

Parses command-line arguments for the EpiAutoGP model.

# Arguments

- `--json-input::String` (required): Path to JSON file containing model input data.
- `--output-dir::String` (required): Directory for saving model outputs.
- `--n-forecast-weeks::Int` (default: 8): Number of weeks to forecast.
- `--n-particles::Int` (default: 24): Number of particles for SMC.
- `--n-mcmc::Int` (default: 100): Number of MCMC steps for GP kernel structure.
- `--n-hmc::Int` (default: 50): Number of HMC steps for GP kernel hyperparameters.
- `--n-forecast-draws::Int` (default: 2000): Number of forecast draws.
- `--transformation::String` (default: "boxcox"): Data transformation type ("boxcox", "positive", "percentage").
- `--smc-data-proportion::Float64` (default: 0.1): Proportion of data used in each SMC step.

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
        "--n-forecast-weeks"
        help = "Number of weeks to forecast"
        arg_type = Int
        default = 8
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
        "--transformation"
        help = "Data transformation type (boxcox, positive, percentage)"
        arg_type = String
        default = "boxcox"
        "--smc-data-proportion"
        help = "Proportion of data used in each SMC step"
        arg_type = Float64
        default = 0.1
    end

    return parse_args(s)
end
