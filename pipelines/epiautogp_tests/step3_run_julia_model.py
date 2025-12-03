#!/usr/bin/env python3
"""
Step 3: Run Julia EpiAutoGP model.

This script takes the epiautogp_input.json created in Step 2 and runs the
Julia EpiAutoGP model to generate forecasts.

Usage:
    python pipelines/epiautogp_tests/step3_run_julia_model.py
"""

from pathlib import Path

# Configuration - must match Steps 1 and 2
TEST_OUTPUT_DIR = Path("pipelines/epiautogp_tests/test_output")
EPIAUTOGP_INPUT_JSON = TEST_OUTPUT_DIR / "epiautogp_input.json"
JULIA_PROJECT_PATH = Path("EpiAutoGP")
MODEL_OUTPUT_DIR = TEST_OUTPUT_DIR / "model_output"

# Model parameters (minimal for testing)
MODEL_PARAMS = {
    "n-forecast-weeks": 4,  # 4 weeks forecast
    "n-particles": 12,  # Reduced for faster testing (default: 24)
    "n-mcmc": 50,  # Reduced for faster testing (default: 100)
    "n-hmc": 25,  # Reduced for faster testing (default: 50)
    "n-forecast-draws": 1000,  # Reduced for faster testing (default: 2000)
}


def main():
    """Run the Julia EpiAutoGP model."""
    print("=" * 70)
    print("EpiAutoGP Integration Test - Step 3: Run Julia Model")
    print("=" * 70)

    # Verify input file exists
    if not EPIAUTOGP_INPUT_JSON.exists():
        print(f"\n✗ ERROR: Input file not found: {EPIAUTOGP_INPUT_JSON}")
        print("\nPlease run Steps 1 and 2 first:")
        print("  python pipelines/epiautogp_tests/step1_generate_test_data.py")
        print("  python pipelines/epiautogp_tests/step2_transform_to_epiautogp.py")
        return 1

    # Create output directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nInput JSON: {EPIAUTOGP_INPUT_JSON}")
    print(f"Output directory: {MODEL_OUTPUT_DIR}")
    print(f"Julia project: {JULIA_PROJECT_PATH}")
    print("\nModel parameters:")
    for key, value in MODEL_PARAMS.items():
        print(f"  {key}: {value}")

    try:
        print("\n" + "=" * 70)
        print("Initializing Julia model...")
        print("=" * 70 + "\n")

        # Add the run.jl script to the parameters
        run_script = JULIA_PROJECT_PATH / "run.jl"
        if not run_script.exists():
            raise FileNotFoundError(f"Julia run script not found: {run_script}")

        print("\n" + "=" * 70)
        print("Running Julia model...")
        print(
            "=" * 70 + "\n"
        )  # Run the model - manually construct command since JuliaModel doesn't handle the script path
        import subprocess

        cmd_parts = [
            "julia",
            f"--project={JULIA_PROJECT_PATH}",
            "--threads=1",
            str(run_script),
            f"--json-input={EPIAUTOGP_INPUT_JSON}",
            f"--output-dir={MODEL_OUTPUT_DIR}",
        ]

        # Add model parameters
        for key, value in MODEL_PARAMS.items():
            cmd_parts.append(f"--{key}={value}")

        cmd_str = " ".join(cmd_parts)
        print(f"Running command: {cmd_str}\n")

        result = subprocess.run(cmd_str, shell=True, capture_output=False)

        if result.returncode != 0:
            raise RuntimeError(f"Julia model failed with exit code {result.returncode}")

        print("\n" + "=" * 70)
        print("✓ Step 3 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nModel outputs saved to: {MODEL_OUTPUT_DIR}")

        # List output files
        output_files = list(MODEL_OUTPUT_DIR.glob("*"))
        if output_files:
            print("\nGenerated files:")
            for f in sorted(output_files):
                print(f"  - {f.name}")

        print("\nNext steps:")
        print("  4. Verify outputs")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Step 3 FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
