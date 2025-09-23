using Test
using ArgParse
using Dates

# Include the parse_arguments function
include("../parse_arguments.jl")

@testset "parse_arguments tests" begin
    
    @testset "required arguments validation" begin
        # Test that parse_arguments function exists and returns an ArgParseSettings
        s = ArgParseSettings()
        @test s isa ArgParseSettings
        
        # Test that the function can be called
        @test parse_arguments isa Function
    end
    
    @testset "argument parsing with mock ARGS" begin
        # Mock command line arguments
        test_args = [
            "--json-input", "/path/to/input.json",
            "--output-dir", "/path/to/output", 
            "--disease", "COVID-19",
            "--location", "CA",
            "--forecast-date", "2024-12-21"
        ]
        
        # Parse arguments by temporarily setting ARGS
        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)
            
            parsed = parse_arguments()
            
            # Test required arguments
            @test parsed["json-input"] == "/path/to/input.json"
            @test parsed["output-dir"] == "/path/to/output"
            @test parsed["disease"] == "COVID-19"
            @test parsed["location"] == "CA"
            @test parsed["forecast-date"] == Date("2024-12-21")
            
            # Test default values
            @test parsed["n-forecast-weeks"] == 4
            @test parsed["n-particles"] == 24
            @test parsed["n-mcmc"] == 100
            @test parsed["n-hmc"] == 50
            @test parsed["n-forecast-draws"] == 2000
            @test parsed["n-redact"] == 1
            
        finally
            # Restore original ARGS
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end
    
    @testset "custom argument values" begin
        # Test with custom values for optional arguments
        test_args = [
            "--json-input", "/test/input.json",
            "--output-dir", "/test/output",
            "--disease", "Influenza", 
            "--location", "NY",
            "--forecast-date", "2024-11-15",
            "--n-forecast-weeks", "6",
            "--n-particles", "48",
            "--n-mcmc", "200",
            "--n-hmc", "100",
            "--n-forecast-draws", "3000",
            "--n-redact", "2"
        ]
        
        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)
            
            parsed = parse_arguments()
            
            @test parsed["disease"] == "Influenza"
            @test parsed["location"] == "NY"
            @test parsed["forecast-date"] == Date("2024-11-15")
            @test parsed["n-forecast-weeks"] == 6
            @test parsed["n-particles"] == 48
            @test parsed["n-mcmc"] == 200
            @test parsed["n-hmc"] == 100
            @test parsed["n-forecast-draws"] == 3000
            @test parsed["n-redact"] == 2
            
        finally
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end
    
    @testset "argument types" begin
        test_args = [
            "--json-input", "/test.json",
            "--output-dir", "/test",
            "--disease", "RSV",
            "--location", "TX", 
            "--forecast-date", "2024-10-01"
        ]
        
        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)
            
            parsed = parse_arguments()
            
            # Test argument types
            @test parsed["json-input"] isa String
            @test parsed["output-dir"] isa String
            @test parsed["disease"] isa String
            @test parsed["location"] isa String
            @test parsed["forecast-date"] isa Date
            @test parsed["n-forecast-weeks"] isa Int
            @test parsed["n-particles"] isa Int
            @test parsed["n-mcmc"] isa Int
            @test parsed["n-hmc"] isa Int
            @test parsed["n-forecast-draws"] isa Int
            @test parsed["n-redact"] isa Int
            
        finally
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end
end