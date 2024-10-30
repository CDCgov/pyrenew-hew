import json
import os
from pathlib import Path

import polars as pl

_disease_map = {
    "COVID-19": "COVID-19/Omicron",
}

_inverse_disease_map = {v: k for k, v in _disease_map.items()}


def process_state_level_data(
    state_level_nssp_data: pl.LazyFrame,
    state_abb,
    disease: str,
    first_training_date,
    state_level_report_date,
) -> pl.DataFrame:
    if state_level_nssp_data is None:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "geo_value": pl.Utf8,
                "disease": pl.Utf8,
                "ed_visits": pl.Float64,
            }
        )

    disease_key = _disease_map.get(disease, disease)

    return (
        state_level_nssp_data.filter(
            pl.col("disease").is_in([disease_key, "Total"]),
            pl.col("metric") == "count_ed_visits",
            pl.col("geo_value") == state_abb,
            pl.col("geo_type") == "state",
            pl.col("reference_date") >= first_training_date,
            pl.col("report_date") == state_level_report_date,
        )
        .select(
            [
                pl.col("reference_date").alias("date"),
                pl.col("geo_value").cast(pl.Utf8),
                pl.col("disease").cast(pl.Utf8),
                pl.col("value").alias("ed_visits"),
            ]
        )
        .with_columns(
            disease=pl.col("disease")
            .cast(pl.Utf8)
            .replace(_inverse_disease_map),
        )
        .sort(["date", "disease"])
        .collect()
    )


def aggregate_facility_level_nssp_to_state(
    facility_level_nssp_data: pl.LazyFrame,
    state_abb: str,
    disease: str,
    first_training_date: str,
) -> pl.DataFrame:
    if facility_level_nssp_data is None:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "geo_value": pl.Utf8,
                "disease": pl.Utf8,
                "ed_visits": pl.Float64,
            }
        )

    disease_key = _disease_map.get(disease, disease)

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
            disease=pl.col("disease")
            .cast(pl.Utf8)
            .replace(_inverse_disease_map),
            geo_value=pl.lit(state_abb).cast(pl.Utf8),
        )
        .rename({"reference_date": "date"})
        .sort(["date", "disease"])
        .select(["date", "geo_value", "disease", "ed_visits"])
        .collect()
    )


def verify_no_date_gaps(df: pl.DataFrame):
    expected_length = df.select(
        dur=((pl.col("date").max() - pl.col("date").min()).dt.total_days() + 1)
    ).to_numpy()[0]
    if not df.height == 2 * expected_length:
        raise ValueError("Data frame appears to have date gaps")


def process_and_save_state(
    state_abb,
    disease,
    report_date,
    state_level_report_date,
    first_training_date,
    last_training_date,
    param_estimates,
    model_batch_dir,
    logger=None,
    facility_level_nssp_data: pl.LazyFrame = None,
    state_level_nssp_data: pl.LazyFrame = None,
    mode="forecast",
) -> None:
    if mode not in ["forecast", "eval"]:
        raise ValueError(
            f"Invalid mode: {mode}. Mode must be 'forecast' or 'eval'."
        )

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
        )
        .with_columns(
            # get estimate from nearest the report date
            diff_from_report=pl.col("reference_date") - report_date
        )
        .abs()
        .filter(pl.col("diff_from_report") == pl.col("diff_from_report").min())
        .collect()
        .get_column("value")
        .to_list()[0]
    )

    right_truncation_offset = (report_date - last_training_date).days

    aggregated_facility_data = aggregate_facility_level_nssp_to_state(
        facility_level_nssp_data=facility_level_nssp_data,
        state_abb=state_abb,
        disease=disease,
        first_training_date=first_training_date,
    )

    state_level_data = process_state_level_data(
        state_level_nssp_data=state_level_nssp_data,
        state_abb=state_abb,
        disease=disease,
        first_training_date=first_training_date,
        state_level_report_date=state_level_report_date,
    )

    if aggregated_facility_data.height > 0:
        first_facility_level_data_date = aggregated_facility_data.get_column(
            "date"
        ).min()
        state_level_data = state_level_data.filter(
            pl.col("date") < first_facility_level_data_date
        )

    data_to_save = (
        pl.concat([state_level_data, aggregated_facility_data])
        .with_columns(
            pl.when(pl.col("date") <= last_training_date)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .alias("data_type"),
        )
        .sort(["date", "disease"])
    )

    verify_no_date_gaps(data_to_save)

    train_disease_ed_visits = (
        data_to_save.filter(
            pl.col("data_type") == "train", pl.col("disease") == disease
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

    data_for_model_fit = {
        "inf_to_hosp_pmf": delay_pmf,
        "generation_interval_pmf": generation_interval_pmf,
        "right_truncation_pmf": right_truncation_pmf,
        "data_observed_disease_hospital_admissions": train_disease_ed_visits,
        "data_observed_total_hospital_admissions": train_total_ed_visits,
        "state_pop": state_pop,
        "right_truncation_offset": right_truncation_offset,
    }

    state_dir = os.path.join(model_batch_dir, state_abb)
    os.makedirs(state_dir, exist_ok=True)
    if logger is not None:
        logger.info(f"Saving {state_abb} to {state_dir}")

    if mode == "forecast":
        data_to_save.write_csv(Path(state_dir, "data.tsv"), sep="\t")
    elif mode == "eval":
        with open(
            Path(state_dir, "data_for_model_fit.json"), "w"
        ) as json_file:
            json.dump(data_for_model_fit, json_file)

    return None
