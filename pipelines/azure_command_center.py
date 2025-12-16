import datetime as dt
import os
import re
from functools import partial
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv
from postprocess_forecast_batches import main as postprocess
from rich import print
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from pipelines.batch.setup_job import main as setup_job

DEFAULT_RNG_KEY = 12345
load_dotenv()
console = Console()

H_DISEASES = {"COVID-19", "Influenza", "RSV"}
E_DISEASES = {"COVID-19", "Influenza", "RSV"}
W_DISEASES = {"COVID-19"}
ALL_DISEASES = H_DISEASES | E_DISEASES | W_DISEASES
# ND: wastewater data not available
# TN: wastewater data unusable (dry sludge)
W_EXCLUDE_DEFAULT = ["US", "TN", "ND"]
# WY: no E data available
E_EXCLUDE_DEFAULT = ["WY"]

today = dt.date.today()
today_str = today.strftime("%Y-%m-%d")
output_subdir = f"{today_str}_forecasts"


def get_env_or_prompt(var_name: str, default: str = "") -> str:
    value = os.getenv(var_name, default)
    if not value:
        prompt = f"Environment variable [bold]{var_name}[/bold] not found. Enter a temporary value"
        value = Prompt.ask(prompt)

    return value


def setup_job_append_id(
    model_letters: str,
    job_id: str,
    pool_id: str,
    model_family: str,
    diseases: str | list[str],
    output_subdir: str | Path = "./",
    additional_forecast_letters: str = "",
    container_image_name: str = "pyrenew-hew",
    container_image_version: str = "latest",
    n_training_days: int = 150,
    exclude_last_n_days: int = 1,
    rng_key: int = DEFAULT_RNG_KEY,
    locations_include: list[str] | None = None,
    locations_exclude: list[str] | None = None,
    test: bool = False,
    append_id: str = "",
):
    updated_job_id = job_id + append_id
    if Confirm.ask(f"Submit job {updated_job_id}?"):
        setup_job(
            model_letters=model_letters,
            job_id=updated_job_id,
            pool_id=pool_id,
            model_family=model_family,
            diseases=diseases,
            output_subdir=output_subdir,
            additional_forecast_letters=additional_forecast_letters,
            container_image_name=container_image_name,
            container_image_version=container_image_version,
            n_training_days=n_training_days,
            exclude_last_n_days=exclude_last_n_days,
            rng_key=rng_key,
            locations_include=locations_include,
            locations_exclude=locations_exclude,
            test=test,
        )


fit_timeseries_e = partial(
    setup_job_append_id,
    model_letters="e",
    job_id="timeseries-e-prod-",
    pool_id="pyrenew-pool",
    model_family="timeseries",
    diseases=E_DISEASES,
    output_subdir=output_subdir,
    locations_exclude=E_EXCLUDE_DEFAULT,
)

fit_pyrenew_e = partial(
    setup_job_append_id,
    model_letters="e",
    job_id="pyrenew-e-prod-",
    pool_id="pyrenew-pool",
    model_family="pyrenew",
    diseases=E_DISEASES,
    output_subdir=output_subdir,
    locations_exclude=E_EXCLUDE_DEFAULT,
)

fit_pyrenew_h = partial(
    setup_job_append_id,
    model_letters="h",
    job_id="pyrenew-h-prod-",
    pool_id="pyrenew-pool",
    model_family="pyrenew",
    diseases=H_DISEASES,
    output_subdir=output_subdir,
)

fit_pyrenew_he = partial(
    setup_job_append_id,
    model_letters="he",
    job_id="pyrenew-he-prod-",
    pool_id="pyrenew-pool",
    model_family="pyrenew",
    diseases=H_DISEASES & E_DISEASES,
    output_subdir=output_subdir,
    locations_exclude=E_EXCLUDE_DEFAULT,
)

fit_pyrenew_hw = partial(
    setup_job_append_id,
    model_letters="hw",
    job_id="pyrenew-hw-prod-",
    pool_id="pyrenew-pool-32gb",
    model_family="pyrenew",
    diseases=H_DISEASES & W_DISEASES,
    output_subdir=output_subdir,
    locations_exclude=W_EXCLUDE_DEFAULT,
)

fit_pyrenew_hew = partial(
    setup_job_append_id,
    model_letters="hew",
    job_id="pyrenew-hew-prod-",
    pool_id="pyrenew-pool-32gb",
    model_family="pyrenew",
    diseases=H_DISEASES & E_DISEASES & W_DISEASES,
    output_subdir=output_subdir,
    locations_exclude=E_EXCLUDE_DEFAULT + W_EXCLUDE_DEFAULT,
)


def ask_about_reruns():
    locations_input = Prompt.ask(
        "Enter locations to include (space-separated, or press Enter for all locations)",
        default="",
    ).strip()
    locations_include = (
        None
        if locations_input == ""
        else [loc.strip() for loc in locations_input.split(" ")]
    )
    e_exclude_last_n_days = IntPrompt.ask(
        "How many days to exclude for E signal?", default=1
    )
    h_exclude_last_n_days = IntPrompt.ask(
        "How many days to exclude for H signal?", default=1
    )
    rng_key = IntPrompt.ask("RNG seed for reproducibility?", default=DEFAULT_RNG_KEY)

    return {
        "locations_include": locations_include,
        "e_exclude_last_n_days": e_exclude_last_n_days,
        "h_exclude_last_n_days": h_exclude_last_n_days,
        "rng_key": rng_key,
    }


def compute_skips(e_exclude_last_n_days: int, h_exclude_last_n_days: int, rng_key: int):
    skip_e = e_exclude_last_n_days == 1 and rng_key == DEFAULT_RNG_KEY
    skip_h = h_exclude_last_n_days == 1 and rng_key == DEFAULT_RNG_KEY
    skip_he = (
        max(e_exclude_last_n_days, h_exclude_last_n_days) == 1
        and rng_key == DEFAULT_RNG_KEY
    )
    return {"skip_e": skip_e, "skip_h": skip_h, "skip_he": skip_he}


def do_timeseries_reruns(
    locations_include: list[str] | None = None,
    e_exclude_last_n_days: int = 1,
    h_exclude_last_n_days: int = 1,
    rng_key: int = DEFAULT_RNG_KEY,  # not used, but kept for interface consistency
    append_id: str = "",
):
    skips = compute_skips(e_exclude_last_n_days, h_exclude_last_n_days, rng_key)

    if skips["skip_e"]:
        print("Skipping Timeseries-E re-fitting due to E")
    else:
        fit_timeseries_e(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=e_exclude_last_n_days,
        )
    if skips["skip_h"]:
        print("Skipping Timeseries-E re-fitting due to H")
    else:
        fit_timeseries_e(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=h_exclude_last_n_days,
        )


def do_pyrenew_reruns(
    locations_include: list[str] | None = None,
    e_exclude_last_n_days: int = 1,
    h_exclude_last_n_days: int = 1,
    rng_key: int = DEFAULT_RNG_KEY,
    append_id: str = "",
):
    he_exclude_last_n_days = max(e_exclude_last_n_days, h_exclude_last_n_days)
    skips = compute_skips(e_exclude_last_n_days, h_exclude_last_n_days, rng_key)

    if skips["skip_e"]:
        print("Skipping PyRenew-E re-fitting")
    else:
        fit_pyrenew_e(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=e_exclude_last_n_days,
            rng_key=rng_key,
        )

    if skips["skip_h"]:
        print("Skipping PyRenew-H re-fitting")
    else:
        fit_pyrenew_h(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=h_exclude_last_n_days,
            rng_key=rng_key,
        )
        fit_pyrenew_hw(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=h_exclude_last_n_days,
            rng_key=rng_key,
        )

    if skips["skip_he"]:
        print("Skipping PyRenew-HE and HEW re-fitting")
    else:
        fit_pyrenew_he(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=he_exclude_last_n_days,
            rng_key=rng_key,
        )

        fit_pyrenew_hew(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=he_exclude_last_n_days,
            rng_key=rng_key,
        )


def get_data_status(
    nssp_etl_path: Path,
    nwss_vintages_path: Path,
    nhsn_target_url: str,
    latest_comprehensive_filename: str = "latest_comprehensive.parquet",
    gold_subdir: str = "gold",
):
    """Get the status of various datasets including update dates and days behind."""
    latest_comprehensive_path = nssp_etl_path / latest_comprehensive_filename
    nssp_gold_files = list((nssp_etl_path / gold_subdir).glob("*.parquet"))
    if not nssp_gold_files:
        raise FileNotFoundError(
            f"No .parquet files found in the directory: {nssp_etl_path / gold_subdir}"
        )
    latest_gold_path = max(nssp_gold_files)

    nssp_gold_update_date = dt.datetime.strptime(
        latest_gold_path.stem, "%Y-%m-%d"
    ).date()

    nwss_gold_dirs = list(nwss_vintages_path.glob("NWSS-ETL-covid-*"))
    if not nwss_gold_dirs:
        raise FileNotFoundError(
            f"No NWSS-ETL-covid-* directories found in the path: {nwss_vintages_path}"
        )
    latest_nwss_path = max(nwss_gold_dirs)

    date_match = re.search(r"\d{4}-\d{2}-\d{2}$", latest_nwss_path.name)
    if not date_match:
        raise ValueError(
            f"Filename does not contain a valid date: {latest_nwss_path.name}"
        )
    nwss_update_date = dt.datetime.strptime(date_match.group(), "%Y-%m-%d").date()

    latest_comprehensive_update_date = (
        pl.read_parquet(latest_comprehensive_path)
        .select(pl.col("report_date").max())
        .item(0, "report_date")
    )

    try:
        response = requests.get(nhsn_target_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        nhsn_data = response.json()
        nhsn_update_date_raw = nhsn_data.get("rowsUpdatedAt")
        if nhsn_update_date_raw is None:
            raise ValueError("Key 'rowsUpdatedAt' not found in NHSN API response.")
        nhsn_update_date = dt.datetime.fromtimestamp(nhsn_update_date_raw).date()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from NHSN API: {e}")
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"Error processing NHSN API response: {e}")

    datasets = {
        "nssp-etl/gold": nssp_gold_update_date,
        "nwss-etl": nwss_update_date,
        "latest_comprehensive": latest_comprehensive_update_date,
        "NHSN API": nhsn_update_date,
    }

    return datasets


def get_status(days_behind):
    if days_behind == 0:
        return Text("✅ Current", style="bold green")
    else:
        return Text("❌ Stale", style="bold red")


def print_data_status(datasets):
    """Print a formatted table showing dataset status."""
    # Move this to the main body with other global variables

    # Create a table
    table = Table(
        title="Dataset Status Report",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Last Updated", style="green")
    table.add_column("Days Behind", style="yellow")
    table.add_column("Status", justify="center")

    # Helper function to get status emoji and color

    for name, update_date in datasets.items():
        days_behind = (today - update_date).days
        table.add_row(
            name,
            str(update_date),
            str(days_behind),
            get_status(days_behind),
        )

    # Print the table
    console.print(table)


def ask_integer_choice(choices):
    """
    Asks the user to select an integer choice from a list of options.

    :param choices: A list of choices to present to the user.
    :return: The index of the selected choice (1-based).
    """
    if not choices:
        raise ValueError("Choices list cannot be empty")
    while True:
        print("\nWhat would you like to do?")
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")

        choice = IntPrompt.ask(
            f"Enter a number between [b]1[/b] and [b]{len(choices)}[/b]"
        )
        if choice >= 1 and choice <= len(choices):
            return choices[choice - 1]
        else:
            print(f"[prompt.invalid]Number must be between 1 and {len(choices)}")


if __name__ == "__main__":
    nssp_etl_path = Path(get_env_or_prompt("NSSP_ETL_PATH"))
    pyrenew_hew_prod_output_path = Path(
        get_env_or_prompt("PYRENEW_HEW_PROD_OUTPUT_PATH")
    )
    nwss_vintages_path = Path(get_env_or_prompt("NWSS_VINTAGES_PATH"))
    nhsn_target_url = "https://data.cdc.gov/api/views/mpgq-jmmr.json"

    choices = [
        "Fit initial Timeseries Models",
        "Fit initial PyRenew-E Models",
        "Fit initial PyRenew-H** models",
        "Rerun Timeseries Models",
        "Rerun PyRenew Models",
        "Postprocess Forecast Batches",
        "Exit",
    ]

    # Get and print data status
    datasets = get_data_status(nssp_etl_path, nwss_vintages_path, nhsn_target_url)
    print_data_status(datasets)

    while True:
        selected_choice = ask_integer_choice(choices)
        current_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if selected_choice == "Exit":
            print("Exiting...")
            break
        elif selected_choice == "Fit initial Timeseries Models":
            fit_timeseries_e(append_id=current_time)
        elif selected_choice == "Fit initial PyRenew-E Models":
            fit_pyrenew_e(append_id=current_time)
        elif selected_choice == "Fit initial PyRenew-H** models":
            fit_pyrenew_h(append_id=current_time)
            fit_pyrenew_he(append_id=current_time)
            fit_pyrenew_hw(append_id=current_time)
            fit_pyrenew_hew(append_id=current_time)
        elif selected_choice == "Rerun Timeseries Models":
            ask_about_reruns_input = ask_about_reruns()
            do_timeseries_reruns(append_id=current_time, **ask_about_reruns_input)
        elif selected_choice == "Rerun PyRenew Models":
            ask_about_reruns_input = ask_about_reruns()
            do_pyrenew_reruns(append_id=current_time, **ask_about_reruns_input)
        elif selected_choice == "Postprocess Forecast Batches":
            postprocess(
                base_forecast_dir=pyrenew_hew_prod_output_path / output_subdir,
                diseases=ALL_DISEASES,
            )

        input("Press enter to continue...")
