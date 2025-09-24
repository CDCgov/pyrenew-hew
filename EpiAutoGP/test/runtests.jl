using Test, EpiAutoGP
using JSON3
using ArgParse
using Dates

# Run all tests in the test directory
include("test_parse_arguments.jl")
include("test_input.jl")

println("All EpiAutoGP tests completed!")
