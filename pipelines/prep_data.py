import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import tempfile
import tomllib
from datetime import datetime
from logging import Logger
from pathlib import Path

import forecasttools
import jax.numpy as jnp
import polars as pl
import polars.selectors as cs
from prep_ww_data import clean_nwss_data, preprocess_ww_data

from pyrenew_hew.utils import approx_lognorm

_disease_map = {
    "COVID-19": "COVID-19/Omicron",
}

_inverse_disease_map = {v: k for k, v in _disease_map.items()}


def get_nhsn(
    start_date: dt.date,
    end_date: dt.date,
    disease: str,
    loc_abb: str,
    temp_dir: Path = None,
    credentials_dict: dict = None,
    local_data_file: Path = None,
) -> pl.DataFrame:
    if local_data_file is None:
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

        loc_abb_for_query = loc_abb if loc_abb != "US" else "USA"

        local_data_file = Path(temp_dir, "nhsn_temp.parquet")
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
                jurisdictions = {py_scalar_to_r_scalar(loc_abb_for_query)}
            ) |>
            dplyr::mutate(weekendingdate = lubridate::as_date(weekendingdate)) |>
            dplyr::mutate(jurisdiction = dplyr::if_else(jurisdiction == "USA", "US",
            jurisdiction
            )) |>
            dplyr::rename(hospital_admissions = {py_scalar_to_r_scalar(columns)}) |>
            dplyr::mutate(hospital_admissions = as.numeric(hospital_admissions)) |>
            forecasttools::write_tabular("{str(local_data_file)}")
            """,
        ]

        result = subprocess.run(r_command)

        if result.returncode != 0:
            raise RuntimeError(
                f"pull_and_save_nhsn: {result.stderr.decode('utf-8')}"
            )
    raw_dat = pl.read_parquet(local_data_file)
    dat = raw_dat.with_columns(
        weekendingdate=pl.col("weekendingdate").cast(pl.Date)
    )
    return dat


def combine_surveillance_data(
    nssp_data: pl.DataFrame,
    nhsn_data: pl.DataFrame,
    disease: str,
    nwss_data: pl.DataFrame = None,
):
    nssp_data_long = nssp_data.unpivot(
        on=["observed_ed_visits", "other_ed_visits"],
        variable_name=".variable",
        index=cs.exclude(["observed_ed_visits", "other_ed_visits"]),
        value_name=".value",
    ).with_columns(pl.lit(None).alias("lab_site_index"))

    nhsn_data_long = (
        nhsn_data.rename(
            {
                "weekendingdate": "date",
                "jurisdiction": "geo_value",
                "hospital_admissions": "observed_hospital_admissions",
            }
        )
        .unpivot(
            on="observed_hospital_admissions",
            index=cs.exclude("observed_hospital_admissions"),
            variable_name=".variable",
            value_name=".value",
        )
        .with_columns(pl.lit(None).alias("lab_site_index"))
    )

    nwss_data_long = (
        nwss_data.rename(
            {
                "log_genome_copies_per_ml": "site_level_log_ww_conc",
                "location": "geo_value",
            }
        )
        .with_columns(pl.lit("train").alias("data_type"))
        .select(
            cs.exclude(
                [
                    "lab",
                    "log_lod",
                    "below_lod",
                    "site",
                    "site_index",
                    "site_pop",
                    "lab_site_name",
                ]
            )
        )
        .unpivot(
            on="site_level_log_ww_conc",
            index=cs.exclude("site_level_log_ww_conc"),
            variable_name=".variable",
            value_name=".value",
        )
        if nwss_data is not None
        else pl.DataFrame()
    )

    combined_dat = (
        pl.concat(
            [nssp_data_long, nhsn_data_long, nwss_data_long],
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
                "lab_site_index",
            ]
        )
    )

    return combined_dat


def aggregate_to_national(
    data: pl.LazyFrame,
    geo_values_to_include: list[str],
    first_date_to_include: dt.date,
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


def process_loc_level_data(
    loc_level_nssp_data: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    first_training_date: dt.date,
    loc_pop_df: pl.DataFrame,
) -> pl.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if loc_level_nssp_data is None:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "geo_value": pl.Utf8,
                "disease": pl.Utf8,
                "ed_visits": pl.Float64,
            }
        )

    disease_key = _disease_map.get(disease, disease)

    if loc_abb == "US":
        locations_to_aggregate = (
            loc_pop_df.filter(pl.col("abb") != "US")
            .get_column("abb")
            .unique()
            .to_list()
        )
        logger.info("Aggregating state-level data to national")
        loc_level_nssp_data = aggregate_to_national(
            loc_level_nssp_data,
            locations_to_aggregate,
            first_training_date,
            national_geo_value="US",
        )

    return (
        loc_level_nssp_data.filter(
            pl.col("disease").is_in([disease_key, "Total"]),
            pl.col("metric") == "count_ed_visits",
            pl.col("geo_value") == loc_abb,
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
        .collect(engine="streaming")
    )


def aggregate_facility_level_nssp_to_loc(
    facility_level_nssp_data: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    first_training_date: str,
    loc_pop_df: pl.DataFrame,
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

    if loc_abb == "US":
        logger.info("Aggregating facility-level data to national")
        locations_to_aggregate = (
            loc_pop_df.filter(pl.col("abb") != "US").get_column("abb").unique()
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
            pl.col("geo_value") == loc_abb,
            pl.col("reference_date") >= first_training_date,
        )
        .group_by(["reference_date", "disease"])
        .agg(pl.col("value").sum().alias("ed_visits"))
        .with_columns(
            disease=pl.col("disease")
            .cast(pl.Utf8)
            .replace(_inverse_disease_map),
            geo_value=pl.lit(loc_abb).cast(pl.Utf8),
        )
        .rename({"reference_date": "date"})
        .sort(["date", "disease"])
        .select(["date", "geo_value", "disease", "ed_visits"])
        .collect()
    )


def get_loc_pop_df():
    return forecasttools.location_table.select(
        pl.col("short_name").alias("abb"),
        pl.col("long_name").alias("name"),
        pl.col("population"),
    )


def generate_epiweekly_data(
    model_run_dir: Path, data_names: str = None
) -> None:
    command = [
        "Rscript",
        "pipelines/generate_epiweekly_data.R",
        f"{model_run_dir}",
    ]
    if data_names is not None:
        command.extend(["--data-names", f"{data_names}"])

    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"generate_epiweekly_data: {result.stderr.decode('utf-8')}"
        )
    return None


def process_and_save_loc(
    loc_abb: str,
    disease: str,
    report_date: dt.date,
    first_training_date: dt.date,
    last_training_date: dt.date,
    model_run_dir: Path,
    param_estimates: pl.LazyFrame = None,
    logger: Logger = None,
    facility_level_nssp_data: pl.LazyFrame = None,
    loc_level_nssp_data: pl.LazyFrame = None,
    loc_level_nwss_data: pl.LazyFrame = None,
    credentials_dict: dict = None,
    nhsn_data_path: Path | str = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if facility_level_nssp_data is None and loc_level_nssp_data is None:
        raise ValueError(
            "Must provide at least one "
            "of facility-level and state-level"
            "NSSP data"
        )

    loc_pop_df = get_loc_pop_df()

    loc_pop = loc_pop_df.filter(pl.col("abb") == loc_abb).item(0, "population")

    right_truncation_offset = (report_date - last_training_date).days - 1
    # First entry of source right truncation PMFs corresponds to reports
    # for ref date = report_date - 1 as of report_date

    aggregated_facility_data = aggregate_facility_level_nssp_to_loc(
        facility_level_nssp_data=facility_level_nssp_data,
        loc_abb=loc_abb,
        disease=disease,
        first_training_date=first_training_date,
        loc_pop_df=loc_pop_df,
    )

    loc_level_data = process_loc_level_data(
        loc_level_nssp_data=loc_level_nssp_data,
        loc_abb=loc_abb,
        disease=disease,
        first_training_date=first_training_date,
        loc_pop_df=loc_pop_df,
    )

    if aggregated_facility_data.height > 0:
        first_facility_level_data_date = aggregated_facility_data.get_column(
            "date"
        ).min()
        loc_level_data = loc_level_data.filter(
            pl.col("date") < first_facility_level_data_date
        )

    nssp_training_data = (
        pl.concat([loc_level_data, aggregated_facility_data])
        .filter(pl.col("date") <= last_training_date)
        .with_columns(pl.lit("train").alias("data_type"))
        .pivot(
            on="disease",
            values="ed_visits",
        )
        .rename({disease: "observed_ed_visits", "Total": "other_ed_visits"})
        .sort("date")
    )

    nhsn_training_data = (
        get_nhsn(
            start_date=first_training_date,
            end_date=last_training_date,
            disease=disease,
            loc_abb=loc_abb,
            credentials_dict=credentials_dict,
            local_data_file=nhsn_data_path,
        )
        .filter(
            (pl.col("weekendingdate") <= last_training_date)
            & (pl.col("weekendingdate") >= first_training_date)
        )  # in testing mode, this isn't guaranteed
        .with_columns(pl.lit("train").alias("data_type"))
    )

    nhsn_step_size = 7

    nwss_training_data = (
        loc_level_nwss_data.to_dict(as_series=False)
        if loc_level_nwss_data is not None
        else None
    )

    if loc_level_nwss_data is None:
        pop_fraction = jnp.array([1])
    else:
        subpop_sizes = (
            loc_level_nwss_data.select(["site_index", "site", "site_pop"])
            .unique()
            .sort("site_pop", descending=True)
            .get_column("site_pop")
            .to_numpy()
        )
        if loc_pop > sum(subpop_sizes):
            pop_fraction = (
                jnp.concatenate(
                    (jnp.array([loc_pop - sum(subpop_sizes)]), subpop_sizes)
                )
                / loc_pop
            )
        else:
            pop_fraction = subpop_sizes / sum(subpop_sizes)

    data_for_model_fit = {
        "loc_pop": loc_pop,
        "right_truncation_offset": right_truncation_offset,
        "pop_fraction": pop_fraction.tolist(),
        "nwss_training_data": nwss_training_data,
        "nssp_training_data": nssp_training_data.to_dict(as_series=False),
        "nhsn_training_data": nhsn_training_data.to_dict(as_series=False),
        "nhsn_step_size": nhsn_step_size,
        "nssp_step_size": 1,
        "nwss_step_size": 1,
    }

    data_dir = Path(model_run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    with open(Path(data_dir, "data_for_model_fit.json"), "w") as json_file:
        json.dump(data_for_model_fit, json_file, default=str)

    combined_training_dat = combine_surveillance_data(
        nssp_data=nssp_training_data,
        nhsn_data=nhsn_training_data,
        nwss_data=loc_level_nwss_data,
        disease=disease,
    )

    if logger is not None:
        logger.info(f"Saving {loc_abb} to {data_dir}")

    combined_training_dat.write_csv(
        Path(data_dir, "combined_training_data.tsv"), separator="\t"
    )
    return None


def get_training_dates(report_date, exclude_last_n_days, n_training_days):
    # + 1 because max date in dataset is report_date - 1
    last_training_date = report_date - dt.timedelta(
        days=exclude_last_n_days + 1
    )
    if last_training_date >= report_date:
        raise ValueError(
            "Last training date must be before the report date. "
            "Got a last training date of {last_training_date} "
            "with a report date of {report_date}."
        )
    first_training_date = last_training_date - dt.timedelta(
        days=n_training_days - 1
    )
    return (last_training_date, first_training_date)


def get_available_reports(
    data_dir: str | Path, glob_pattern: str = "*.parquet"
):
    return [
        datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(data_dir).glob(glob_pattern)
    ]


def main(
    disease: str,
    loc: str,
    report_date: str,
    facility_level_nssp_data_dir: Path | str,
    state_level_nssp_data_dir: Path | str,
    nwss_data_dir: Path | str,
    output_dir: Path | str,
    n_training_days: int,
    credentials_path: Path = None,
    nhsn_data_path: Path | str = None,
    exclude_last_n_days: int = 0,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if credentials_path is not None:
        cp = Path(credentials_path)
        if not cp.suffix.lower() == ".toml":
            raise ValueError(
                "Credentials file must have the extension "
                "'.toml' (not case-sensitive). Got "
                f"{cp.suffix}"
            )
        logger.info(f"Reading in credentials from {cp}...")
        with open(cp, "rb") as fp:
            credentials_dict = tomllib.load(fp)
    else:
        logger.info("No credentials file given. Will proceed without one.")
        credentials_dict = None

    report_date = dt.datetime.strptime(report_date, "%Y-%m-%d").date()
    logger.info(f"Report date: {report_date}")
    (last_training_date, first_training_date) = get_training_dates(
        report_date,
        exclude_last_n_days,
        n_training_days,
    )

    logger.info(f"First training date {first_training_date}")
    logger.info(f"last training date: {last_training_date}")

    available_facility_level_reports = get_available_reports(
        facility_level_nssp_data_dir
    )
    available_loc_level_reports = get_available_reports(
        state_level_nssp_data_dir
    )
    first_available_loc_report = min(available_loc_level_reports)
    last_available_loc_report = max(available_loc_level_reports)

    if report_date in available_loc_level_reports:
        loc_report_date = report_date
    elif report_date > last_available_loc_report:
        loc_report_date = last_available_loc_report
    elif report_date > first_available_loc_report:
        raise ValueError(
            "Dataset appear to be missing some state-level "
            f"reports. First entry is {first_available_loc_report}, "
            f"last is {last_available_loc_report}, but no entry "
            f"for {report_date}"
        )
    else:
        raise ValueError(
            "Requested report date is earlier than the first "
            "state-level vintage. This is not currently supported"
        )

    if loc_report_date is not None:
        logger.info(f"Using location-level data as of: {loc_report_date}")

    facility_level_nssp_data, loc_level_nssp_data = None, None

    if report_date in available_facility_level_reports:
        logger.info("Facility level data available for the given report date")
        facility_datafile = f"{report_date}.parquet"
        facility_level_nssp_data = pl.scan_parquet(
            Path(facility_level_nssp_data_dir, facility_datafile)
        )
    if loc_report_date in available_loc_level_reports:
        logger.info("location-level data available for the given report date.")
        loc_datafile = f"{loc_report_date}.parquet"
        loc_level_nssp_data = pl.scan_parquet(
            Path(state_level_nssp_data_dir, loc_datafile)
        )
    if facility_level_nssp_data is None and loc_level_nssp_data is None:
        raise ValueError(
            f"No data available for the requested report date {report_date}"
        )

    nwss_data_disease_map = {
        "COVID-19": "covid",
        "Influenza": "flu",
    }

    def get_available_nwss_reports(
        data_dir: str | Path,
        glob_pattern: str = f"NWSS-ETL-{nwss_data_disease_map[disease]}-",
    ):
        return [
            datetime.strptime(
                f.stem.removeprefix(glob_pattern), "%Y-%m-%d"
            ).date()
            for f in Path(data_dir).glob(f"{glob_pattern}*")
        ]

    available_nwss_reports = get_available_nwss_reports(nwss_data_dir)
    if report_date in available_nwss_reports:
        nwss_data_raw = pl.scan_parquet(
            Path(
                nwss_data_dir,
                f"NWSS-ETL-{nwss_data_disease_map[disease]}-{report_date}",
                "bronze.parquet",
            )
        )
        nwss_data_cleaned = clean_nwss_data(nwss_data_raw).filter(
            (pl.col("location") == loc)
            & (pl.col("date") >= first_training_date)
        )
        loc_level_nwss_data = preprocess_ww_data(nwss_data_cleaned.collect())
    else:
        raise ValueError(
            "NWSS data not available for the requested report date "
            f"{report_date}"
        )

    model_batch_dir_name = (
        f"{disease.lower()}_r_{report_date}_f_"
        f"{first_training_date}_t_{last_training_date}"
    )

    model_batch_dir = Path(output_dir, model_batch_dir_name)

    model_run_dir = Path(model_batch_dir, "model_runs", loc)
    os.makedirs(model_run_dir, exist_ok=True)

    logger.info(f"Processing {loc}")
    process_and_save_loc(
        loc_abb=loc,
        disease=disease,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        loc_level_nwss_data=loc_level_nwss_data,
        report_date=report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        model_run_dir=model_run_dir,
        logger=logger,
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly_data(model_run_dir)

    logger.info(
        "Data preparation complete."
        f" for location {loc}, and "
        f"report date {report_date}."
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create fit data for disease modeling."
    )
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
            "Two-letter USPS abbreviation for the location to fit"
            "(e.g. 'AK', 'AL', 'AZ', etc.)."
        ),
    )

    parser.add_argument(
        "--report-date",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Report date in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--facility-level-nssp-data-dir",
        type=Path,
        default=Path("private_data", "nssp_etl_gold"),
        help=(
            "Directory in which to look for facility-level NSSP ED visit data"
        ),
    )

    parser.add_argument(
        "--state-level-nssp-data-dir",
        type=Path,
        default=Path("private_data", "nssp_state_level_gold"),
        help=(
            "Directory in which to look for state-level NSSP ED visit data."
        ),
    )

    parser.add_argument(
        "--nwss-data-dir",
        type=Path,
        default=Path("private_data", "nwss_vintages"),
        help=("Directory in which to look for NWSS data."),
    )

    parser.add_argument(
        "--credentials-path",
        type=Path,
        help=("Path to a TOML file containing credentials such as API keys."),
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
        "--exclude-last-n-days",
        type=int,
        default=0,
        help=(
            "Optionally exclude the final n days of available training "
            "data (Default: 0, i.e. exclude no available data"
        ),
    )

    parser.add_argument(
        "--nhsn-data-path",
        type=Path,
        help=("Path to local NHSN data (for local testing)"),
        default=None,
    )

    args = parser.parse_args()
    main(**vars(args))
