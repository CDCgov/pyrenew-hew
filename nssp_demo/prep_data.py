import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


def aggregate_facility_level_nssp_to_state(
    facility_level_nssp_data: pl.LazyFrame,
    state_abb: str,
    disease: str,
    first_training_date: str,
    last_training_date: str,
) -> pl.DataFrame:
    disease_map = {
        "COVID-19": "COVID-19/Omicron",
    }
    disease_key = disease_map.get(disease, disease)

    return (
        facility_level_nssp_data.filter(
            pl.col("disease").is_in([disease_key, "Total"]),
            pl.col("metric") == "count_ed_visits",
            pl.col("geo_value") == state_abb,
            pl.col("reference_date") >= first_training_date,
        )
        .group_by(["reference_date", "disease"])
        .agg(pl.col("value").sum().alias("ed_visits"))
        .with_columns(
            pl.when(pl.col("reference_date") <= last_training_date)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .alias("data_type"),
            disease=pl.col("disease")
            .cast(pl.Utf8)
            .replace({v: k for k, v in disease_map.items()})
            .cast(pl.Categorical),
        )
        .rename({"reference_date": "date"})
        .sort(["date", "disease"])
        .collect()
    )


def process_and_save_state(
    state_abb,
    disease,
    report_date,
    first_training_date,
    last_training_date,
    param_estimates,
    model_batch_dir,
    logger=None,
    facility_level_nssp_data: pl.LazyFrame = None,
    state_level_nssp_data: pl.LazyFrame = None,
) -> None:
    if facility_level_nssp_data is None and state_level_nssp_data is None:
        raise ValueError(
            "Must provide at least one "
            "of facility-level and state-level"
            "NSSP data"
        )

    facts = pl.read_csv(
        "https://raw.githubusercontent.com/k5cents/usa/"
        "refs/heads/master/data-raw/facts.csv"
    )
    states = pl.read_csv(
        "https://raw.githubusercontent.com/k5cents/usa/"
        "refs/heads/master/data-raw/states.csv"
    )

    state_pop_df = facts.join(states, on="name").select(
        ["abb", "name", "population"]
    )

    state_pop = (
        state_pop_df.filter(pl.col("abb") == state_abb)
        .get_column("population")
        .to_list()[0]
    )

    generation_interval_pmf = (
        param_estimates.filter(
            (pl.col("geo_value").is_null())
            & (pl.col("disease") == disease)
            & (pl.col("parameter") == "generation_interval")
            & (pl.col("end_date").is_null())  # most recent estimate
        )
        .collect()
        .get_column("value")
        .to_list()[0]
    )

    delay_pmf = (
        param_estimates.filter(
            (pl.col("geo_value").is_null())
            & (pl.col("disease") == disease)
            & (pl.col("parameter") == "delay")
            & (pl.col("end_date").is_null())  # most recent estimate
        )
        .collect()
        .get_column("value")
        .to_list()[0]
    )

    right_truncation_pmf = (
        param_estimates.filter(
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
        .collect()
        .get_column("value")
        .to_list()[0]
    )

    right_truncation_offset = (report_date - last_training_date).days

    data_to_save = aggregate_facility_level_nssp_to_state(
        facility_level_nssp_data=facility_level_nssp_data,
        state_abb=state_abb,
        disease=disease,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
    )

    train_disease_ed_visits = (
        data_to_save.filter(
            pl.col("data_type") == "train", pl.col("disease") == disease
        )
        .get_column("ed_visits")
        .to_list()
    )

    test_disease_ed_visits = (
        data_to_save.filter(
            pl.col("data_type") == "test",
            pl.col("disease") == disease,
        )
        .get_column("ed_visits")
        .to_list()
    )

    train_total_ed_visits = (
        data_to_save.filter(
            pl.col("data_type") == "train", pl.col("disease") == "Total"
        )
        .get_column("ed_visits")
        .to_list()
    )

    test_total_ed_visits = (
        data_to_save.filter(
            pl.col("data_type") == "test", pl.col("disease") == "Total"
        )
        .get_column("ed_visits")
        .to_list()
    )

    data_for_model_fit = {
        "inf_to_hosp_pmf": delay_pmf,
        "generation_interval_pmf": generation_interval_pmf,
        "right_truncation_pmf": right_truncation_pmf,
        "data_observed_disease_hospital_admissions": train_disease_ed_visits,
        "data_observed_disease_hospital_admissions_test": test_disease_ed_visits,
        "data_observed_total_hospital_admissions": train_total_ed_visits,
        "data_observed_total_hospital_admissions_test": test_total_ed_visits,
        "state_pop": state_pop,
        "right_truncation_offset": right_truncation_offset,
    }

    state_dir = os.path.join(model_batch_dir, state_abb)
    os.makedirs(state_dir, exist_ok=True)

    if logger is not None:
        logger.info(f"Saving {state_abb} to {state_dir}")
    data_to_save.write_csv(Path(state_dir, "data.csv"))

    with open(Path(state_dir, "data_for_model_fit.json"), "w") as json_file:
        json.dump(data_for_model_fit, json_file)

    return None


def main(
    disease,
    report_date,
    facility_level_nssp_data_dir,
    state_level_nssp_data_dir,
    param_data_dir,
    output_data_dir,
    training_day_offset,
    n_training_days,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if report_date == "latest":
        report_date = max(
            f.stem
            for f in Path(facility_level_nssp_data_dir).glob("*.parquet")
        )

    report_date = datetime.strptime(report_date, "%Y-%m-%d").date()

    logger.info(f"Report date: {report_date}")

    last_training_date = report_date - timedelta(days=training_day_offset + 1)
    # +1 because max date in dataset is report_date - 1
    first_training_date = last_training_date - timedelta(
        days=n_training_days - 1
    )

    datafile = f"{report_date}.parquet"
    facility_level_nssp_data = pl.scan_parquet(
        Path(facility_level_nssp_data_dir, datafile)
    )

    state_level_nssp_data = None
    if state_level_nssp_data_dir is not None:
        state_level_nssp_data = pl.scan_parquet(
            Path(facility_level_nssp_data_dir, datafile)
        )

    param_estimates = pl.scan_parquet(Path(param_data_dir, "prod.parquet"))

    excluded_states = ["GU", "MO", "WY"]

    all_states = (
        facility_level_nssp_data.select(pl.col("geo_value").unique())
        .filter(~pl.col("geo_value").is_in(excluded_states))
        .collect()
        .get_column("geo_value")
        .to_list()
    )
    all_states.sort()

    model_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    model_batch_dir = Path(output_data_dir, model_dir_name)

    os.makedirs(model_batch_dir, exist_ok=True)

    for state_abb in all_states:
        logger.info(f"Processing {state_abb}")
        process_and_save_state(
            state_abb=state_abb,
            disease=disease,
            facility_level_nssp_data=facility_level_nssp_data,
            state_level_nssp_data=state_level_nssp_data,
            report_date=report_date,
            first_training_date=first_training_date,
            last_training_date=last_training_date,
            param_estimates=param_estimates,
            model_batch_dir=model_batch_dir,
            logger=logger,
        )
    logger.info("Data preparation complete.")

    return None


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
    "--report-date",
    type=str,
    default="latest",
    help="Report date in YYYY-MM-DD format or latest (default: latest)",
)

parser.add_argument(
    "--facility-level-nssp-data-dir",
    type=Path,
    default=Path("private_data", "nssp_etl_gold"),
    help=(
        "Directory in which to look for facility-level NSSP " "ED visit data"
    ),
)

parser.add_argument(
    "--state-level-nssp-data-dir",
    type=Path,
    default=None,
    help=(
        "Directory in which to look for state-level NSSP "
        "ED visit data (Default: None, which implies using "
        "only facility-level data"
    ),
)

parser.add_argument(
    "--param-data-dir",
    type=Path,
    default=Path("private_data", "prod_param_estimates"),
    help=(
        "Directory in which to look for parameter estimates"
        "such as delay PMFs."
    ),
)

parser.add_argument(
    "--output-data-dir",
    type=Path,
    default="private_data",
    help="Directory in which to save output data.",
)

parser.add_argument(
    "--training-day-offset",
    type=int,
    default=7,
    help=(
        "Number of days before the reference day "
        "to use as test data (default: 7)"
    ),
)

parser.add_argument(
    "--n-training-days",
    type=int,
    default=90,
    help="Number of training days (default: 90)",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
