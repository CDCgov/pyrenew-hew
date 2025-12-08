"""Shared command-line arguments for forecast pipelines."""

import argparse
from pathlib import Path


def add_common_forecast_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments shared across forecast pipelines."""
    parser.add_argument(
        "--disease",
        type=str,
        required=True,
        help="Disease to model (e.g., COVID-19, Influenza, RSV).",
    )

    parser.add_argument(
        "--loc",
        type=str,
        required=True,
        help=(
            "Two-letter USPS abbreviation for the location to fit "
            "(e.g. 'AK', 'AL', 'AZ', etc.)."
        ),
    )

    parser.add_argument(
        "--report-date",
        type=str,
        default="latest",
        help="Report date in YYYY-MM-DD format or latest (default: latest).",
    )

    parser.add_argument(
        "--facility-level-nssp-data-dir",
        type=Path,
        default=Path("private_data", "nssp_etl_gold"),
        help="Directory in which to look for facility-level NSSP ED visit data.",
    )

    parser.add_argument(
        "--state-level-nssp-data-dir",
        type=Path,
        default=Path("private_data", "nssp_state_level_gold"),
        help="Directory in which to look for state-level NSSP ED visit data.",
    )

    parser.add_argument(
        "--param-data-dir",
        type=Path,
        default=Path("private_data", "prod_param_estimates"),
        help="Directory in which to look for parameter estimates such as delay PMFs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="private_data",
        help="Directory in which to save output.",
    )

    parser.add_argument(
        "--n-training-days",
        type=int,
        default=180,
        help="Number of training days (default: 180).",
    )

    parser.add_argument(
        "--n-forecast-days",
        type=int,
        default=28,
        help=(
            "Number of days ahead to forecast relative to the "
            "report date (default: 28)."
        ),
    )

    parser.add_argument(
        "--exclude-last-n-days",
        type=int,
        default=0,
        help=(
            "Optionally exclude the final n days of available training "
            "data (default: 0, i.e. exclude no available data)."
        ),
    )

    parser.add_argument(
        "--credentials-path",
        type=Path,
        help="Path to a TOML file containing credentials such as API keys.",
    )

    parser.add_argument(
        "--eval-data-path",
        type=Path,
        help="Path to a parquet file containing comprehensive truth data.",
    )

    parser.add_argument(
        "--nhsn-data-path",
        type=Path,
        default=None,
        help="Path to local NHSN data (for local testing).",
    )
