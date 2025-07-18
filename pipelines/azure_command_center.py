import datetime as dt
import os
from functools import partial
from pathlib import Path

import polars as pl
import requests
from rich import print
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .batch.setup_job import main as setup_job
from .postprocess_forecast_batches import main as postprocess

# to do: work with specific diseases
DISEASES = ["COVID-19"]  # not forecasting flu currently
W_EXCLUDE_DEFAULT = ["US", "NY"]

today = dt.date.today()
today_str = today.strftime("%Y-%m-%d")
output_subdir = f"{today_str}_forecasts"


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
    locations_include: list[str] | None = None,
    locations_exclude: list[str] | None = None,
    test: bool = False,
    append_id: str = "",
):
    updated_job_id = job_id + append_id
    print(f"Submitting job: {updated_job_id}")
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
    diseases=DISEASES,
    output_subdir=output_subdir,
)

fit_pyrenew_e = partial(
    setup_job_append_id,
    model_letters="e",
    job_id="pyrenew-e-prod-",
    pool_id="pyrenew-pool",
    model_family="pyrenew",
    diseases=DISEASES,
    output_subdir=output_subdir,
)

fit_pyrenew_h = partial(
    setup_job_append_id,
    model_letters="h",
    job_id="pyrenew-h-prod-",
    pool_id="pyrenew-pool",
    model_family="pyrenew",
    diseases=DISEASES,
    output_subdir=output_subdir,
)

fit_pyrenew_he = partial(
    setup_job_append_id,
    model_letters="he",
    job_id="pyrenew-he-prod-",
    pool_id="pyrenew-pool",
    model_family="pyrenew",
    diseases=DISEASES,
    output_subdir=output_subdir,
)

fit_pyrenew_hw = partial(
    setup_job_append_id,
    model_letters="hw",
    job_id="pyrenew-hw-prod-",
    pool_id="pyrenew-pool-32gb",
    model_family="pyrenew",
    diseases=DISEASES,
    output_subdir=output_subdir,
    locations_exclude=W_EXCLUDE_DEFAULT,
)

fit_pyrenew_hew = partial(
    setup_job_append_id,
    model_letters="hew",
    job_id="pyrenew-hew-prod-",
    pool_id="pyrenew-pool-32gb",
    model_family="pyrenew",
    diseases=DISEASES,
    output_subdir=output_subdir,
    locations_exclude=W_EXCLUDE_DEFAULT,
)


def ask_about_reruns():
    locations_input = input(
        "Enter locations to include (space-separated, or press Enter for all locations): "
    ).strip()
    locations_include = (
        None
        if locations_input == ""
        else [loc.strip() for loc in locations_input.split(" ")]
    )
    while True:
        try:
            e_input = input(
                "How many days to exclude for E signal? (default: 1): "
            ).strip()
            e_exclude_last_n_days = 1 if e_input == "" else int(e_input)
            break
        except ValueError:
            print("Please enter a valid integer.")

    while True:
        try:
            h_input = input(
                "How many days to exclude for H signal? (default: 1): "
            ).strip()
            h_exclude_last_n_days = 1 if h_input == "" else int(h_input)
            break
        except ValueError:
            print("Please enter a valid integer.")

    return {
        "locations_include": locations_include,
        "e_exclude_last_n_days": e_exclude_last_n_days,
        "h_exclude_last_n_days": h_exclude_last_n_days,
    }


def do_timeseries_reruns(
    locations_include: list[str] | None = None,
    e_exclude_last_n_days: int = 1,
    h_exclude_last_n_days: int = 1,
    append_id: str = "",
):
    if e_exclude_last_n_days == 1:
        print("Skipping Timeseries-E re-fitting due to E")
    else:
        fit_timeseries_e(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=e_exclude_last_n_days,
        )
    if h_exclude_last_n_days == 1:
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
    append_id: str = "",
):
    he_exclude_last_n_days = max(e_exclude_last_n_days, h_exclude_last_n_days)
    if e_exclude_last_n_days == 1:
        print("Skipping PyRenew-E re-fitting")
    else:
        fit_pyrenew_e(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=e_exclude_last_n_days,
        )

    if h_exclude_last_n_days == 1:
        print("Skipping PyRenew-H re-fitting")
    else:
        fit_pyrenew_h(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=h_exclude_last_n_days,
        )
        fit_pyrenew_hw(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=h_exclude_last_n_days,
        )

    if he_exclude_last_n_days == 1:
        print("Skipping PyRenew-HE and HEW re-fitting")
    else:
        fit_pyrenew_he(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=he_exclude_last_n_days,
        )

        fit_pyrenew_hew(
            append_id=append_id,
            locations_include=locations_include,
            exclude_last_n_days=he_exclude_last_n_days,
        )


nssp_etl_path = Path(os.environ["nssp_etl_path"])
pyrenew_hew_prod_output_path = Path(os.environ["pyrenew_hew_prod_output_path"])
latest_comprehensive_path = nssp_etl_path / "latest_comprehensive.parquet"
latest_gold_path = max((nssp_etl_path / "gold").glob("*.parquet"))

gold_update_date = dt.datetime.strptime(
    latest_gold_path.stem, "%Y-%m-%d"
).date()


latest_comprehensive_update_date = (
    pl.read_parquet(latest_comprehensive_path)
    .select(pl.col("report_date").max())
    .item(0, "report_date")
)


nhsn_update_date_raw = (
    requests.get("https://data.cdc.gov/api/views/mpgq-jmmr.json")
    .json()
    .get("rowsUpdatedAt")
)
nhsn_update_date = dt.datetime.fromtimestamp(nhsn_update_date_raw).date()


console = Console()

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

# Calculate days behind and add rows to table
datasets = [
    ("nssp-etl/gold", gold_update_date),
    ("latest_comprehensive", latest_comprehensive_update_date),
    ("NHSN API", nhsn_update_date),
]


# Helper function to get status emoji and color
def get_status(days_behind):
    if days_behind == 0:
        return Text("✅ Current", style="bold green")
    else:
        return Text("❌ Stale", style="bold red")


for name, update_date in datasets:
    days_behind = (today - update_date).days
    table.add_row(
        name,
        str(update_date),
        str(days_behind),
        get_status(days_behind),
    )

# Print the table
console.print(table)


# Ask user for action
print("\nWhat would you like to do?")
choices = [
    "Fit initial Timeseries Models",
    "Fit initial PyRenew-E Models",
    "Fit initial PyRenew-H** models",
    "Rerun Timeseries Models",
    "Rerun PyRenew Models",
    "Postprocess Forecast Batches",
    "Exit",
]


while True:
    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")
    choice = input(f"\nEnter your choice (1-{len(choices)}): ")
    current_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(choices):
            raise IndexError()
        selected_choice = choices[choice_idx]

        # You can add specific logic here based on the selected choice
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
            do_timeseries_reruns(
                append_id=current_time, **ask_about_reruns_input
            )
        elif selected_choice == "Rerun PyRenew Models":
            ask_about_reruns_input = ask_about_reruns()
            do_pyrenew_reruns(append_id=current_time, **ask_about_reruns_input)
        elif selected_choice == "Postprocess Forecast Batches":
            postprocess(
                base_forecast_dir=pyrenew_hew_prod_output_path / output_subdir,
                diseases=DISEASES,
            )
        else:
            print(f"Executing: {selected_choice}")
        input("Press any key to continue...")
    except (ValueError, IndexError):
        print(
            f"Invalid input. Please enter a number between 1-{len(choices)}."
        )
