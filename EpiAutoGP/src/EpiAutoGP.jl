module EpiAutoGP
using NowcastAutoGP # Core modeling package
using CSV, Dates, JSON3, StructTypes # Data handling packages
using ArgParse # Command-line argument parsing
using Statistics # For modeling functions
using TidierData # For DataFrame and data manipulation

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
       forecast_with_epiautogp

# Export output functions
export create_hubverse_table,
       save_model_outputs,
       load_json_data,
       prepare_epiautogp_data,
       run_epiautogp_model

include("parse_arguments.jl") # Function to parse command line arguments
include("input.jl")           # Functions to load and process input JSON data
include("modelling.jl")        # Main modeling and forecasting functions
include("output.jl")           # Functions for generating hubverse outputs

end
