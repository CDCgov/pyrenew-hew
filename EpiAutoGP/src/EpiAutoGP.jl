module EpiAutoGP
using NowcastAutoGP # Core modeling package
using CSV, Dates, JSON3, StructTypes, TidierData # Data handling packages
using ArgParse # Command-line argument parsing

# Export command line argument parsing
export parse_arguments

# Export input data structures and functions
export EpiAutoGPInput,
       validate_input,
       read_data,
       read_and_validate_data,
       safe_read_data,
       validate_and_report,
       create_sample_input

include("parse_arguments.jl") # Function to parse command line arguments
include("input.jl")           # Functions to load and process input JSON data

end
