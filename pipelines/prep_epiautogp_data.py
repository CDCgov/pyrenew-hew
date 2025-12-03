import json
from pathlib import Path
from typing import Optional

import polars as pl


def convert_to_epiautogp_format(
    input_json_path: str,
    output_json_path: str,
    target: str = "nhsn",
    disease: str = "COVID-19",
    location: str = "US",
    forecast_date: Optional[str] = None,
    nowcast_reports_path: Optional[str] = None,
) -> dict:
    """
    Convert `data_for_model_fit.json` aimed at PyRenewHEWData to EpiAutoGP input format.

    Args:
        input_json_path: Path to data_for_model_fit.json created by prep_data.py
        output_json_path: Path where EpiAutoGP JSON will be saved
        target: Data source to use - either "nhsn" (hospital admissions) or "nssp" (ED visits)
        disease: Disease name (COVID-19, Influenza, or RSV)
        location: Geographic location code (e.g., "US", "CA", "NY")
        forecast_date: Forecast date (YYYY-MM-DD). If None, uses the last date in the data
        nowcast_reports_path: Path to nowcast samples file. If None, no nowcast is included

    Returns:
        dict: The EpiAutoGP input structure
    """
    print(f"Reading PyRenew data from {input_json_path}")

    # Read the PyRenew JSON file
    with open(input_json_path, "r") as f:
        pyrenew_data = json.load(f)

    # Select the appropriate data source
    if target.lower() == "nhsn":
        if "nhsn_training_data" not in pyrenew_data:
            raise ValueError("NHSN training data not found in input JSON")

        # Convert NHSN data to DataFrame
        nhsn_data = pyrenew_data["nhsn_training_data"]
        df = pl.DataFrame(nhsn_data)

        # Sort by date
        df = df.sort("weekendingdate")

        # Extract dates and reports
        dates = df["weekendingdate"].to_list()
        reports = df["hospital_admissions"].to_list()

        print("Using NHSN hospital admissions data")
        print(f"Date range: {min(dates)} to {max(dates)}")
        print(f"Number of data points: {len(dates)}")

    elif target.lower() == "nssp":
        if "nssp_training_data" not in pyrenew_data:
            raise ValueError("NSSP training data not found in input JSON")

        # Convert NSSP data to DataFrame
        nssp_data = pyrenew_data["nssp_training_data"]
        df = pl.DataFrame(nssp_data)

        # Sort by date
        df = df.sort("date")

        # Extract dates and reports (using observed_ed_visits)
        dates = df["date"].to_list()
        reports = df["observed_ed_visits"].to_list()

        print("Using NSSP ED visits data")
        print(f"Date range: {min(dates)} to {max(dates)}")
        print(f"Number of data points: {len(dates)}")

    else:
        raise ValueError(f"Invalid target '{target}'. Must be 'nhsn' or 'nssp'")

    # Set forecast_date to last date if not provided
    if forecast_date is None:
        forecast_date = max(dates)
        print(f"Using last date as forecast_date: {forecast_date}")

    # Handle nowcast data
    nowcast_dates = []
    nowcast_reports = []

    if nowcast_reports_path is not None:
        print(f"Loading nowcast data from {nowcast_reports_path}")
        with open(nowcast_reports_path, "r") as f:
            nowcast_data = json.load(f)
        nowcast_dates = nowcast_data.get("nowcast_dates", [])
        nowcast_reports = nowcast_data.get("nowcast_reports", [])
        print(
            f"Loaded {len(nowcast_dates)} nowcast dates with {len(nowcast_reports)} sample realizations"
        )
    else:
        print("No nowcast data provided - using empty nowcast")

    # Create the EpiAutoGP input structure
    epiautogp_input = {
        "dates": dates,
        "reports": reports,
        "pathogen": disease,
        "location": location,
        "target": target.lower(),
        "forecast_date": forecast_date,
        "nowcast_dates": nowcast_dates,
        "nowcast_reports": nowcast_reports,
    }

    # Write to JSON file
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(epiautogp_input, f, indent=2)

    print(f"\nSuccessfully created EpiAutoGP input JSON: {output_json_path}")
    print(f"Pathogen: {disease}")
    print(f"Location: {location}")
    print(f"Target: {target}")
    print(f"Forecast date: {forecast_date}")
    print(f"Historical data points: {len(dates)}")
    print(f"Nowcast dates: {len(nowcast_dates)}")

    return epiautogp_input
