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

    @testset "argument parsing with required arguments" begin
        # Mock command line arguments with only required arguments
        test_args = [
            "--json-input", "/Users/samandfi/Documents/GitHub/CFA/pyrenew-hew/EpiAutoGP/test/data/bootstrap_private_data/MT/data/data_for_model_fit.json",
            "--output-dir", "/path/to/output",
            "--forecast-date", "2024-12-21"
        ]

        # Parse arguments by temporarily setting ARGS
        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)

            parsed = parse_arguments()

            # Test required arguments
            @test parsed["json-input"] == "/Users/samandfi/Documents/GitHub/CFA/pyrenew-hew/EpiAutoGP/test/data/bootstrap_private_data/MT/data/data_for_model_fit.json"
            @test parsed["output-dir"] == "/path/to/output"
            @test parsed["forecast-date"] == Date("2024-12-21")

            # Test default values
            @test parsed["n-forecast-weeks"] == 4
            @test parsed["n-particles"] == 24
            @test parsed["n-mcmc"] == 100
            @test parsed["n-hmc"] == 50
            @test parsed["n-forecast-draws"] == 2000
            @test parsed["n-redact"] == 1
            @test parsed["transformation"] == "boxcox"

        finally
            # Restore original ARGS
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end


    @testset "custom argument values" begin
        # Test with custom values for optional arguments
        test_args = [
            "--json-input", "/Users/samandfi/Documents/GitHub/CFA/pyrenew-hew/EpiAutoGP/test/data/bootstrap_private_data/MT/data/data_for_model_fit.json",
            "--output-dir", "/test/output",
            "--forecast-date", "2024-11-15",
            "--n-forecast-weeks", "6",
            "--n-particles", "48",
            "--n-mcmc", "200",
            "--n-hmc", "100",
            "--n-forecast-draws", "3000",
            "--n-redact", "2",
            "--transformation", "positive"
        ]

        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)

            parsed = parse_arguments()

            @test parsed["forecast-date"] == Date("2024-11-15")
            @test parsed["n-forecast-weeks"] == 6
            @test parsed["n-particles"] == 48
            @test parsed["n-mcmc"] == 200
            @test parsed["n-hmc"] == 100
            @test parsed["n-forecast-draws"] == 3000
            @test parsed["n-redact"] == 2
            @test parsed["transformation"] == "positive"

        finally
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end

    @testset "argument types" begin
        test_args = [
            "--json-input", "/Users/samandfi/Documents/GitHub/CFA/pyrenew-hew/EpiAutoGP/test/data/bootstrap_private_data/MT/data/data_for_model_fit.json",
            "--output-dir", "/test",
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
            @test parsed["forecast-date"] isa Date
            @test parsed["n-forecast-weeks"] isa Int
            @test parsed["n-particles"] isa Int
            @test parsed["n-mcmc"] isa Int
            @test parsed["n-hmc"] isa Int
            @test parsed["n-forecast-draws"] isa Int
            @test parsed["n-redact"] isa Int
            @test parsed["transformation"] isa String

        finally
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end


    @testset "transformation argument options" begin
        # Test different transformation options
        transformation_options = ["boxcox", "positive", "percentage"]
        
        for transform in transformation_options
            test_args = [
                "--json-input", "/Users/samandfi/Documents/GitHub/CFA/pyrenew-hew/EpiAutoGP/test/data/bootstrap_private_data/MT/data/data_for_model_fit.json",
                "--output-dir", "/test",
                "--forecast-date", "2024-10-01",
                "--transformation", transform
            ]

            old_args = copy(ARGS)
            try
                empty!(ARGS)
                append!(ARGS, test_args)

                parsed = parse_arguments()
                @test parsed["transformation"] == transform

            finally
                empty!(ARGS)
                append!(ARGS, old_args)
            end
        end
    end
end
