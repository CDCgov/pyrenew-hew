import os
import pathlib
from datetime import timedelta

import duckdb
import polars as pl
import pyarrow.parquet as pq

disease_map = {
    "COVID-19": "COVID-19/Omicron",
    "Influenza": "Influenza",
    "RSV": "RSV",
}
disease = "COVID-19"

import argparse
import json
from datetime import datetime

parser = argparse.ArgumentParser(
    description="Create fit data for disease modeling."
)
parser.add_argument(
    "--disease",
    type=str,
    required=True,
    help="Disease to model (e.g., COVID-19, Influenza, RSV)",
)
parser.add_argument(
    "--report_date",
    type=lambda d: datetime.strptime(d, "%Y-%m-%d").date(),
    # default=(datetime.now()).strftime("%Y-%m-%d"),
    default="2024-10-17",
    help="Report date in YYYY-MM-DD format (default: yesterday)",
)
parser.add_argument(
    "--training_day_offset",
    type=int,
    default=7,
    help="Number of days before the reference day to use as test data (default: 7)",
)
parser.add_argument(
    "--n_training_days",
    type=int,
    default=90,
    help="Number of training days (default: 90)",
)

args = parser.parse_args()

disease = args.disease
report_date = args.report_date
training_day_offset = args.training_day_offset
n_training_days = args.n_training_days

last_training_date = report_date - timedelta(days=training_day_offset + 1)
# +1 because max date in dataset is report_date - 1
first_training_date = last_training_date - timedelta(days=n_training_days - 1)

nssp_data = duckdb.arrow(pq.read_table(f"private_data/{report_date}.parquet"))
nnh_estimates = pl.from_arrow(pq.read_table("private_data/prod.parquet"))


generation_interval_pmf = (
    nnh_estimates.filter(
        (pl.col("geo_value").is_null())
        & (pl.col("disease") == disease)
        & (pl.col("parameter") == "generation_interval")
        & (pl.col("end_date").is_null())  # most recent estimate
    )
    .get_column("value")
    .to_list()[0]
)

delay_pmf = (
    nnh_estimates.filter(
        (pl.col("geo_value").is_null())
        & (pl.col("disease") == disease)
        & (pl.col("parameter") == "delay")
        & (pl.col("end_date").is_null())  # most recent estimate
    )
    .get_column("value")
    .to_list()[0]
)

excluded_states = ["GU", "MO", "WY"]
all_states = (
    nssp_data.unique("geo_value")
    .filter(f"geo_value NOT IN {excluded_states}")
    .order("geo_value")
    .pl()["geo_value"]
    .to_list()
)

facts = pl.read_csv(
    "https://raw.githubusercontent.com/k5cents/usa/refs/heads/master/data-raw/facts.csv"
)
states = pl.read_csv(
    "https://raw.githubusercontent.com/k5cents/usa/refs/heads/master/data-raw/states.csv"
)

state_pop_df = facts.join(states, on="name").select(
    ["abb", "name", "population"]
)

for state_abb in all_states:
    print(f"Processing {state_abb}")
    data_to_save = duckdb.sql(
        f"""
        SELECT report_date, reference_date, SUM(value) AS ED_admissions,
        CASE WHEN reference_date <= '{last_training_date}'
            THEN 'train'
            ELSE 'test' END AS data_type
        FROM nssp_data
        WHERE disease = '{disease_map[disease]}' AND metric = 'count_ed_visits'
        AND geo_value = '{state_abb}'
        and reference_date >= '{first_training_date}'
        GROUP BY report_date, reference_date
        ORDER BY report_date, reference_date
        """
    )
    # why not count_admitted_ed_visits ?

    data_to_save_pl = data_to_save.pl()

    actual_first_date = data_to_save_pl["reference_date"].min()
    actual_last_date = data_to_save_pl["reference_date"].max()

    state_pop = (
        state_pop_df.filter(pl.col("abb") == state_abb)
        .get_column("population")
        .to_list()[0]
    )

    right_truncation_pmf = (
        nnh_estimates.filter(
            (pl.col("geo_value") == state_abb)
            & (pl.col("disease") == disease)
            & (pl.col("parameter") == "right_truncation")
            & (pl.col("end_date").is_null())
            & (
                pl.col("reference_date") <= report_date
            )  # estimates nearest the report date
        )
        .filter(
            pl.col("reference_date") == pl.col("reference_date").max()
        )  # estimates nearest the report date
        .get_column("value")
        .to_list()[0]
    )

    train_ed_admissions = (
        data_to_save_pl.filter(pl.col("data_type") == "train")
        .get_column("ED_admissions")
        .to_list()
    )
    test_ed_admissions = (
        data_to_save_pl.filter(pl.col("data_type") == "test")
        .get_column("ED_admissions")
        .to_list()
    )

    data_for_model_fit = {
        "inf_to_hosp_pmf": delay_pmf,
        "generation_interval_pmf": generation_interval_pmf,
        "right_truncation_pmf": right_truncation_pmf,
        "data_observed_hospital_admissions": train_ed_admissions,
        "test_ed_admissions": test_ed_admissions,
        "state_pop": state_pop,
    }

    model_folder_name = f"{disease.lower()}_r_{report_date}_f_{actual_first_date}_l_{actual_last_date}_t_{last_training_date}"

    model_folder = pathlib.Path("private_data", model_folder_name)
    os.makedirs(model_folder, exist_ok=True)
    state_folder = pathlib.Path(model_folder, state_abb)
    os.makedirs(state_folder, exist_ok=True)
    print(f"Saving {state_abb}")
    data_to_save.to_csv(str(pathlib.Path(state_folder, "data.csv")))

    with open(
        pathlib.Path(state_folder, "data_for_model_fit.json"), "w"
    ) as json_file:
        json.dump(data_for_model_fit, json_file)
