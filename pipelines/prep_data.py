import datetime
import json
import logging
import os
import subprocess
import tempfile
from logging import Logger
from pathlib import Path

import forecasttools
import jax.numpy as jnp
import numpy as np
import polars as pl
import polars.selectors as cs

_disease_map = {
    "COVID-19": "COVID-19/Omicron",
}

_inverse_disease_map = {v: k for k, v in _disease_map.items()}


def get_nhsn(
    start_date: datetime.date,
    end_date: datetime.date,
    disease: str,
    state_abb: str,
    temp_dir: Path = None,
    credentials_dict: dict = None,
) -> None:
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    if credentials_dict is None:
        credentials_dict = dict()

    def py_scalar_to_r_scalar(py_scalar):
        if py_scalar is None:
            return "NULL"
        return f"'{str(py_scalar)}'"

    disease_nhsn_key = {
        "COVID-19": "totalconfc19newadm",
        "Influenza": "totalconfflunewadm",
    }

    columns = disease_nhsn_key[disease]

    state_abb_for_query = state_abb if state_abb != "US" else "USA"

    temp_file = Path(temp_dir, "nhsn_temp.parquet")
    api_key_id = credentials_dict.get(
        "nhsn_api_key_id", os.getenv("NHSN_API_KEY_ID")
    )
    api_key_secret = credentials_dict.get(
        "nhsn_api_key_secret", os.getenv("NHSN_API_KEY_SECRET")
    )

    r_command = [
        "Rscript",
        "-e",
        f"""
        forecasttools::pull_nhsn(
            api_key_id = {py_scalar_to_r_scalar(api_key_id)},
            api_key_secret = {py_scalar_to_r_scalar(api_key_secret)},
            start_date = {py_scalar_to_r_scalar(start_date)},
            end_date = {py_scalar_to_r_scalar(end_date)},
            columns = {py_scalar_to_r_scalar(columns)},
            jurisdictions = {py_scalar_to_r_scalar(state_abb_for_query)}
        ) |>
        dplyr::mutate(weekendingdate = lubridate::as_date(weekendingdate)) |>
        dplyr::mutate(jurisdiction = dplyr::if_else(jurisdiction == "USA", "US",
          jurisdiction
        )) |>
        dplyr::rename(hospital_admissions = {py_scalar_to_r_scalar(columns)}) |>
        dplyr::mutate(hospital_admissions = as.numeric(hospital_admissions)) |>
        arrow::write_parquet("{str(temp_file)}")
        """,
    ]

    result = subprocess.run(r_command)

    if result.returncode != 0:
        raise RuntimeError(f"pull_and_save_nhsn: {result.stderr}")
    raw_dat = pl.read_parquet(temp_file)
    dat = raw_dat.with_columns(
        weekendingdate=pl.col("weekendingdate").cast(pl.Date)
    )
    return dat


def combine_nssp_and_nhsn(
    nssp_data: pl.DataFrame,
    nhsn_data: pl.DataFrame,
    disease: str,
):
    count_type_dict = {
        disease: "observed_ed_visits",
        "Total": "other_ed_visits",
    }

    nssp_data_long = (
        nssp_data.rename({"disease": "count_type"})
        .unpivot(
            on="ed_visits",
            index=cs.exclude(["ed_visits"]),
            variable_name="drop_me",
            value_name=".value",
        )
        .with_columns(
            pl.col("count_type").replace(count_type_dict).alias(".variable")
        )
        .select(cs.exclude(["count_type", "drop_me"]))
    )

    nhsn_data_long = nhsn_data.rename(
        {
            "weekendingdate": "date",
            "jurisdiction": "geo_value",
            "hospital_admissions": "observed_hospital_admissions",
        }
    ).unpivot(
        on="observed_hospital_admissions",
        index=cs.exclude("observed_hospital_admissions"),
        variable_name=".variable",
        value_name=".value",
    )

    combined_dat = (
        pl.concat(
            [nssp_data_long, nhsn_data_long],
            how="diagonal_relaxed",
        )
        .with_columns(pl.lit(disease).alias("disease"))
        .sort(["date", "geo_value", ".variable"])
        .select(
            [
                "date",
                "geo_value",
                "disease",
                "data_type",
                ".variable",
                ".value",
            ]
        )
    )

    return combined_dat


def aggregate_to_national(
    data: pl.LazyFrame,
    geo_values_to_include,
    first_date_to_include: datetime.date,
    national_geo_value="US",
):
    assert national_geo_value not in geo_values_to_include
    return (
        data.filter(
            pl.col("geo_value").is_in(geo_values_to_include),
            pl.col("reference_date") >= first_date_to_include,
        )
        .group_by(["disease", "metric", "geo_type", "reference_date"])
        .agg(geo_value=pl.lit(national_geo_value), value=pl.col("value").sum())
    )


def process_state_level_data(
    state_level_nssp_data: pl.LazyFrame,
    state_abb: str,
    disease: str,
    first_training_date: datetime.date,
    state_pop_df: pl.DataFrame,
) -> pl.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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

    if state_abb == "US":
        locations_to_aggregate = (
            state_pop_df.filter(pl.col("abb") != "US")
            .get_column("abb")
            .unique()
        )
        logger.info("Aggregating state-level data to national")
        state_level_nssp_data = aggregate_to_national(
            state_level_nssp_data,
            locations_to_aggregate,
            first_training_date,
            national_geo_value="US",
        )

    return (
        state_level_nssp_data.filter(
            pl.col("disease").is_in([disease_key, "Total"]),
            pl.col("metric") == "count_ed_visits",
            pl.col("geo_value") == state_abb,
            pl.col("geo_type") == "state",
            pl.col("reference_date") >= first_training_date,
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
        .collect(streaming=True)
    )


def aggregate_facility_level_nssp_to_state(
    facility_level_nssp_data: pl.LazyFrame,
    state_abb: str,
    disease: str,
    first_training_date: str,
    state_pop_df: pl.DataFrame,
) -> pl.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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

    if state_abb == "US":
        logger.info("Aggregating facility-level data to national")
        locations_to_aggregate = (
            state_pop_df.filter(pl.col("abb") != "US")
            .get_column("abb")
            .unique()
        )
        facility_level_nssp_data = aggregate_to_national(
            facility_level_nssp_data,
            locations_to_aggregate,
            first_training_date,
            national_geo_value="US",
        )

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
        .collect(streaming=True)
        # setting streaming = True explicitly
        # avoids an `Option::unwrap()` on a `None` value
        # error. Cause of error not known but presumably
        # related to how parquets are processed.
    )


def verify_no_date_gaps(df: pl.DataFrame):
    expected_length = df.select(
        dur=((pl.col("date").max() - pl.col("date").min()).dt.total_days() + 1)
    ).to_numpy()[0]
    if not df.height == 2 * expected_length:
        raise ValueError("Data frame appears to have date gaps")


def get_state_pop_df():
    return forecasttools.location_table.select(
        pl.col("short_name").alias("abb"),
        pl.col("long_name").alias("name"),
        pl.col("population"),
    )


def get_pmfs(param_estimates: pl.LazyFrame, state_abb: str, disease: str):
    generation_interval_pmf = (
        param_estimates.filter(
            (pl.col("geo_value").is_null())
            & (pl.col("disease") == disease)
            & (pl.col("parameter") == "generation_interval")
            & (pl.col("end_date").is_null())  # most recent estimate
        )
        .collect(streaming=True)
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
        .collect(streaming=True)
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
        .filter(pl.col("reference_date") == pl.col("reference_date").max())
        .collect(streaming=True)
        .get_column("value")
        .to_list()[0]
    )

    return (generation_interval_pmf, delay_pmf, right_truncation_pmf)


def process_and_save_state(
    state_abb: str,
    disease: str,
    report_date: datetime.date,
    first_training_date: datetime.date,
    last_training_date: datetime.date,
    param_estimates: pl.LazyFrame,
    model_run_dir: Path,
    logger: Logger = None,
    facility_level_nssp_data: pl.LazyFrame = None,
    state_level_nssp_data: pl.LazyFrame = None,
    state_level_nwss_data: pl.LazyFrame = None,
    credentials_dict: dict = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if facility_level_nssp_data is None and state_level_nssp_data is None:
        raise ValueError(
            "Must provide at least one "
            "of facility-level and state-level"
            "NSSP data"
        )

    state_pop_df = get_state_pop_df()

    state_pop = state_pop_df.filter(pl.col("abb") == state_abb).item(
        0, "population"
    )

    (generation_interval_pmf, delay_pmf, right_truncation_pmf) = get_pmfs(
        param_estimates=param_estimates, state_abb=state_abb, disease=disease
    )

    right_truncation_offset = (report_date - last_training_date).days

    aggregated_facility_data = aggregate_facility_level_nssp_to_state(
        facility_level_nssp_data=facility_level_nssp_data,
        state_abb=state_abb,
        disease=disease,
        first_training_date=first_training_date,
        state_pop_df=state_pop_df,
    )

    state_level_data = process_state_level_data(
        state_level_nssp_data=state_level_nssp_data,
        state_abb=state_abb,
        disease=disease,
        first_training_date=first_training_date,
        state_pop_df=state_pop_df,
    )

    if aggregated_facility_data.height > 0:
        first_facility_level_data_date = aggregated_facility_data.get_column(
            "date"
        ).min()
        state_level_data = state_level_data.filter(
            pl.col("date") < first_facility_level_data_date
        )

    nssp_training_data = (
        pl.concat([state_level_data, aggregated_facility_data])
        .filter(pl.col("date") <= last_training_date)
        .with_columns(pl.lit("train").alias("data_type"))
        .sort(["date", "disease"])
    )

    verify_no_date_gaps(nssp_training_data)

    nhsn_training_data = get_nhsn(
        start_date=first_training_date,
        end_date=last_training_date,
        disease=disease,
        state_abb=state_abb,
        credentials_dict=credentials_dict,
    ).with_columns(pl.lit("train").alias("data_type"))

    nssp_training_dates = (
        nssp_training_data.get_column("date").unique().to_list()
    )
    nhsn_training_dates = (
        nhsn_training_data.get_column("weekendingdate").unique().to_list()
    )

    nhsn_first_date_index = next(
        i
        for i, x in enumerate(nssp_training_dates)
        if x == min(nhsn_training_dates)
    )
    nhsn_step_size = 7

    train_disease_ed_visits = (
        nssp_training_data.filter(pl.col("disease") == disease)
        .get_column("ed_visits")
        .to_list()
    )

    train_total_ed_visits = (
        nssp_training_data.filter(pl.col("disease") == "Total")
        .get_column("ed_visits")
        .to_list()
    )

    train_disease_hospital_admissions = nhsn_training_data.get_column(
        "hospital_admissions"
    ).to_list()

    data_observed_disease_wastewater = (
        state_level_nwss_data.to_dict(as_series=False)
        if state_level_nwss_data is not None
        else None
    )

    if state_level_nwss_data is None:
        pop_fraction = jnp.array([1])
    else:
        subpop_sizes = (
            state_level_nwss_data.select(["site_index", "site", "site_pop"])
            .unique()["site_pop"]
            .to_numpy()
        )
        if state_pop > sum(subpop_sizes):
            pop_fraction = (
                [state_pop - sum(subpop_sizes)] + subpop_sizes
            ) / state_pop
        else:
            pop_fraction = subpop_sizes / state_pop

    data_for_model_fit = {
        "inf_to_ed_pmf": delay_pmf,
        "generation_interval_pmf": generation_interval_pmf,
        "right_truncation_pmf": right_truncation_pmf,
        "data_observed_disease_ed_visits": train_disease_ed_visits,
        "data_observed_total_ed_visits": train_total_ed_visits,
        "data_observed_disease_hospital_admissions": train_disease_hospital_admissions,
        "nssp_training_dates": nssp_training_dates,
        "nhsn_training_dates": nhsn_training_dates,
        "nhsn_first_date_index": nhsn_first_date_index,
        "nhsn_step_size": nhsn_step_size,
        "state_pop": state_pop,
        "right_truncation_offset": right_truncation_offset,
        "data_observed_disease_wastewater": data_observed_disease_wastewater,
        "pop_fraction": pop_fraction.tolist(),
    }

    data_dir = Path(model_run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    with open(Path(data_dir, "data_for_model_fit.json"), "w") as json_file:
        json.dump(data_for_model_fit, json_file, default=str)

    combined_training_dat = combine_nssp_and_nhsn(
        nssp_data=nssp_training_data,
        nhsn_data=nhsn_training_data,
        disease=disease,
    )

    if logger is not None:
        logger.info(f"Saving {state_abb} to {data_dir}")

    combined_training_dat.write_csv(
        Path(data_dir, "combined_training_data.tsv"), separator="\t"
    )
    return None
