module EpiAutoGP
using NowcastAutoGP # Core modeling package
using CSV, Dates, JSON3, StructTypes # Data handling packages
using ArgParse # Command-line argument parsing
using Statistics # For modeling functions

# Export command line argument parsing
export parse_arguments

# Export input data structures and functions
export EpiAutoGPInput,
       validate_input,
       read_data,
       read_and_validate_data

# Export modeling functions
export prepare_for_modelling,
       fit_base_model,
       forecast_with_epiautogp,
       run_epiautogp_pipeline

include("parse_arguments.jl") # Function to parse command line arguments
include("input.jl")           # Functions to load and process input JSON data
include("modelling.jl")        # Main modeling and forecasting functions

end
