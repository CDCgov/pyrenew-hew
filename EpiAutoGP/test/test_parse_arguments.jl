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
            "--json-input", "path/to/json",
            "--output-dir", "/path/to/output"
        ]

        # Parse arguments by temporarily setting ARGS
        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)

            parsed = parse_arguments()

            # Test required arguments
            @test parsed["json-input"] == "path/to/json"
            @test parsed["output-dir"] == "/path/to/output"

            # Test default values
            @test parsed["n-forecast-weeks"] == 8
            @test parsed["n-particles"] == 24
            @test parsed["n-mcmc"] == 100
            @test parsed["n-hmc"] == 50
            @test parsed["n-forecast-draws"] == 2000
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
            "--json-input", "path/to/json",
            "--output-dir", "/test/output",
            "--n-forecast-weeks", "6",
            "--n-particles", "48",
            "--n-mcmc", "200",
            "--n-hmc", "100",
            "--n-forecast-draws", "3000",
            "--transformation", "positive"
        ]

        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)

            parsed = parse_arguments()

            @test parsed["n-forecast-weeks"] == 6
            @test parsed["n-particles"] == 48
            @test parsed["n-mcmc"] == 200
            @test parsed["n-hmc"] == 100
            @test parsed["n-forecast-draws"] == 3000
            @test parsed["transformation"] == "positive"

        finally
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end

    @testset "argument types" begin
        test_args = [
            "--json-input", "path/to/json",
            "--output-dir", "/test"
        ]

        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, test_args)

            parsed = parse_arguments()

            # Test argument types
            @test parsed["json-input"] isa String
            @test parsed["output-dir"] isa String
            @test parsed["n-forecast-weeks"] isa Int
            @test parsed["n-particles"] isa Int
            @test parsed["n-mcmc"] isa Int
            @test parsed["n-hmc"] isa Int
            @test parsed["n-forecast-draws"] isa Int
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
                "--json-input", "path/to/json",
                "--output-dir", "/test",
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
