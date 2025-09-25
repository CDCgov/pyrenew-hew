#!/usr/bin/env python3
"""
Script to create EpiAutoGP JSON input from vintaged NHSN data.

This script filters the CSV data for a specific report_date and creates
a JSON file in the EpiAutoGPInput format required by the Julia EpiAutoGP model.
"""

import json
import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path


def create_epiautogp_input(csv_path: str, report_date: str, output_path: str, normal_mean: float = 0.1, normal_std: float = 0.03, n_samples: int = 100):
    """
    Create EpiAutoGP JSON input from vintaged NHSN CSV data.
    
    Args:
        csv_path: Path to the vintaged_us_nhsn_data.csv file
        report_date: Report date to filter for (e.g., "2025-08-16")
        output_path: Path where the JSON file will be saved
    """
    print(f"Reading CSV data from {csv_path}")
    
    # Read the CSV file using polars
    df = pl.read_csv(csv_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns}")
    
    # Filter for the specific report_date
    filtered_df = df.filter(pl.col("report_date") == report_date)
    
    print(f"Filtered data shape for report_date {report_date}: {filtered_df.shape}")
    
    if filtered_df.height == 0:
        raise ValueError(f"No data found for report_date {report_date}")
    
    # Sort by reference_date to ensure chronological order
    filtered_df = filtered_df.sort("reference_date")
    
    # Extract the data we need
    dates = filtered_df["reference_date"].to_list()
    reports = filtered_df["confirm"].to_list()
    
    print(f"Date range: {min(dates)} to {max(dates)}")
    print(f"Number of data points: {len(dates)}")
    print(f"Report values range: {min(reports)} to {max(reports)}")
    
    # Create nowcast for the last reference date
    print(f"\n=== Creating Nowcast ===")
    last_date = dates[-1]  # Most recent reference date
    last_report = reports[-1]  # Most recent report value
    
    print(f"Last reference date: {last_date}")
    print(f"Last report value: {last_report}")
    
    # Generate 100 nowcast samples using lognormal multiplicative errors
    # LogNormal with mean=0.1, std=0.03 in log space
    np.random.seed(42)  # For reproducible results
    normal_mean = 0.1
    normal_std = 0.03
    n_samples = 100
    
    # Generate normal random numbers and exp-transform to get lognormal multipliers
    normal_samples = np.random.normal(normal_mean, normal_std, n_samples)
    lognormal_multipliers = np.exp(normal_samples)
    
    # Apply multipliers to the last report to get nowcast samples
    nowcast_samples = [float(last_report * multiplier) for multiplier in lognormal_multipliers]
    
    print(f"Generated {n_samples} nowcast samples")
    print(f"Multiplier range: {min(lognormal_multipliers):.4f} to {max(lognormal_multipliers):.4f}")
    print(f"Nowcast sample range: {min(nowcast_samples):.1f} to {max(nowcast_samples):.1f}")
    print(f"Mean nowcast value: {np.mean(nowcast_samples):.1f}")
    
    # Create nowcast_reports as 100 vectors, each of length 1 (for the 1 nowcast date)
    # Each vector represents one realization/sample across all nowcast dates
    nowcast_reports = [[sample] for sample in nowcast_samples]
    
    # Create the EpiAutoGPInput structure
    epiautogp_input = {
        "dates": dates,
        "reports": reports,
        "pathogen": "COVID-19",
        "location": "US", 
        "target": "nhsn",
        "forecast_date": report_date,
        "nowcast_dates": [last_date],  # Single nowcast date (most recent)
        "nowcast_reports": nowcast_reports  # One vector with 100 samples
    }
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(epiautogp_input, f, indent=2)
    
    print(f"Successfully created EpiAutoGP input JSON: {output_path}")
    print(f"Data contains {len(dates)} time points from {min(dates)} to {max(dates)}")
    print(f"Nowcast created for {last_date} with {len(nowcast_samples)} samples")
    
    return epiautogp_input


def main():
    """Main function to create the EpiAutoGP input JSON."""
    # Set file paths
    csv_path = "vintaged_us_nhsn_data.csv"
    report_date = "2025-08-16"
    output_path = "epiautogp_input_2025-08-16.json"
    
    # Default nowcast multiplier parameters
    normal_mean = 0.1
    normal_std = 0.03
    n_samples = 100

    # Check if CSV file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Create the JSON input
    result = create_epiautogp_input(csv_path, report_date, output_path, normal_mean, normal_std, n_samples)
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Pathogen: {result['pathogen']}")
    print(f"Location: {result['location']}")
    print(f"Target: {result['target']}")
    print(f"Forecast date: {result['forecast_date']}")
    print(f"Number of historical data points: {len(result['dates'])}")
    print(f"Date range: {min(result['dates'])} to {max(result['dates'])}")
    print(f"Report values: min={min(result['reports'])}, max={max(result['reports'])}")
    print(f"Nowcast dates: {len(result['nowcast_dates'])} ({result['nowcast_dates']})")
    if len(result['nowcast_reports']) > 0:
        # Each vector represents one realization across all nowcast dates
        first_realization = result['nowcast_reports'][0]
        all_values = [vec[0] for vec in result['nowcast_reports']]  # Extract first (and only) value from each vector
        print(f"Nowcast samples: {len(result['nowcast_reports'])} realizations, each with {len(first_realization)} values, range {min(all_values):.1f}-{max(all_values):.1f}")


if __name__ == "__main__":
    main()