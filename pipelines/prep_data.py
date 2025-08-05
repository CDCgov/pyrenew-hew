import argparse
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import tempfile
import tomllib
from logging import Logger
from pathlib import Path

import forecasttools
import jax.numpy as jnp
import polars as pl
import polars.selectors as cs
import tomli_w
from prep_ww_data import clean_nwss_data, preprocess_ww_data

from pyrenew_hew.utils import approx_lognorm

_disease_map = {
    "COVID-19": "COVID-19/Omicron",
}

_inverse_disease_map = {v: k for k, v in _disease_map.items()}

nwss_data_disease_map = {
    "COVID-19": "covid",
    "Influenza": "flu",
}


def get_available_nwss_reports(
    data_dir: str | Path,
    disease: str,
):
    glob_pattern = f"NWSS-ETL-{nwss_data_disease_map[disease]}-"
    return [
        dt.datetime.strptime(
            f.stem.removeprefix(glob_pattern), "%Y-%m-%d"
        ).date()
        for f in Path(data_dir).glob(f"{glob_pattern}*")
    ]


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


def _validate_and_extract(
    df: pl.DataFrame,
    parameter_name: str,
) -> list:
    df = df.filter(pl.col("parameter") == parameter_name).collect()
    if df.height != 1:
        error_msg = f"Expected exactly one {parameter_name} parameter row, but found {df.height}"
        logging.error(error_msg)
        if df.height > 0:
            logging.error(f"Found rows: {df}")
        raise ValueError(error_msg)
    return df.item(0, "value").to_list()


def get_pmfs(
    param_estimates: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    as_of: dt.date = None,
    reference_date: dt.date = None,
    right_truncation_required: bool = True,
) -> dict[str, list]:
    """
    Filter and extract probability mass functions (PMFs) from
    parameter estimates LazyFrame based on location, disease
    and date filters.

    This function queries a LazyFrame containing epidemiological
    parameters and returns a dictionary of three PMFs:
    delay, generation interval, and right truncation.

    Parameters
    ----------
    param_estimates: pl.LazyFrame
        A LazyFrame containing parameter data with columns
        including 'disease', 'parameter', 'value', 'geo_value',
        'start_date', 'end_date', and 'reference_date'.

    loc_abb : str
        Location abbreviation (geo_value) to filter
        right truncation parameters.

    disease : str
        Name of the disease.

    as_of : dt.datetime.date, optional
        Date for which parameters must be valid
        (start_date <= as_of <= end_date). Defaults
        to the most recent estimates.

    reference_date : dt.datetime.date, optional
        The reference date for right truncation estimates.
        Defaults to as_of value. Selects the most recent estimate
        with reference_date <= this value.

    right_truncation_required : bool, optional
        If False, allows extraction of other pmfs if
        right_truncation estimate is missing

    Returns
    -------
    dict[str, list]
        A dictionary containing three PMF arrays:
        - 'generation_interval_pmf': Generation interval distribution
        - 'delay_pmf': Delay distribution
        - 'right_truncation_pmf': Right truncation distribution

    Raises
    ------
    ValueError
        If exactly one row is not found for any of the required parameters.

    Notes
    -----
    The function applies specific filtering logic for each parameter type:
    - For delay and generation_interval: filters by disease,
      parameter name, and validity date range.
    - For right_truncation: additionally filters by location.
    """
    min_as_of = dt.date(1000, 1, 1)
    max_as_of = dt.date(3000, 1, 1)
    as_of = as_of or max_as_of
    reference_date = reference_date or as_of

    filtered_estimates = (
        param_estimates.with_columns(
            pl.col("start_date").fill_null(min_as_of),
            pl.col("end_date").fill_null(max_as_of),
        )
        .filter(pl.col("disease") == disease)
        .filter(
            pl.col("start_date") <= as_of,
            pl.col("end_date") >= as_of,
        )
    )

    generation_interval_pmf = _validate_and_extract(
        filtered_estimates, "generation_interval"
    )

    delay_pmf = _validate_and_extract(filtered_estimates, "delay")

    # ensure 0 first entry; we do not model the possibility
    # of a zero infection-to-recorded-admission delay in Pyrenew-HEW
    delay_pmf[0] = 0.0
    delay_pmf = jnp.array(delay_pmf)
    delay_pmf = delay_pmf / delay_pmf.sum()
    delay_pmf = delay_pmf.tolist()

    right_truncation_df = filtered_estimates.filter(
        pl.col("geo_value") == loc_abb
    ).filter(pl.col("reference_date") == pl.col("reference_date").max())

    if (
        right_truncation_df.collect().height == 0
        and not right_truncation_required
    ):
        right_truncation_pmf = [1]
    else:
        right_truncation_pmf = _validate_and_extract(
            right_truncation_df, "right_truncation"
        )

    return {
        "generation_interval_pmf": generation_interval_pmf,
        "delay_pmf": delay_pmf,
        "right_truncation_pmf": right_truncation_pmf,
    }


def process_and_save_loc_data(
    loc_abb: str,
    disease: str,
    report_date: dt.date,
    first_training_date: dt.date,
    last_training_date: dt.date,
    model_run_dir: Path,
    logger: Logger = None,
    facility_level_nssp_data: pl.LazyFrame = None,
    loc_level_nssp_data: pl.LazyFrame = None,
    loc_level_nwss_data: pl.LazyFrame = None,
    credentials_dict: dict = None,
    nhsn_data_path: Path | str = None,
) -> None:
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
        if not loc_level_nwss_data.is_empty()
        else None
    )

    data_for_model_fit = {
        "loc_pop": loc_pop,
        "right_truncation_offset": right_truncation_offset,
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


def process_and_save_loc_param(
    loc_abb,
    disease,
    loc_level_nwss_data,
    param_data_dir,
    model_run_dir,
) -> None:
    loc_pop = (
        get_loc_pop_df().filter(pl.col("abb") == loc_abb).item(0, "population")
    )

    if loc_level_nwss_data.is_empty():
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

    param_estimates = pl.scan_parquet(Path(param_data_dir, "prod.parquet"))
    pmfs = get_pmfs(
        param_estimates=param_estimates,
        loc_abb=loc_abb,
        disease=disease,
        right_truncation_required=False,
    )

    inf_to_hosp_admit_lognormal_loc, inf_to_hosp_admit_lognormal_scale = (
        approx_lognorm(
            jnp.array(pmfs["delay_pmf"])[1:],  # only fit the non-zero delays
            loc_guess=0,
            scale_guess=0.5,
        )
    )

    model_params = {
        "population_size": loc_pop,
        "pop_fraction": pop_fraction.tolist(),
        "generation_interval_pmf": pmfs["generation_interval_pmf"],
        "right_truncation_pmf": pmfs["right_truncation_pmf"],
        "inf_to_hosp_admit_lognormal_loc": inf_to_hosp_admit_lognormal_loc,
        "inf_to_hosp_admit_lognormal_scale": inf_to_hosp_admit_lognormal_scale,
        "inf_to_hosp_admit_pmf": pmfs["delay_pmf"],
    }
    with open(Path(model_run_dir, "model_params.json"), "w") as json_file:
        json.dump(model_params, json_file, default=str)

    return None


def copy_and_record_priors(priors_path: Path, model_run_dir: Path):
    metadata_file = Path(model_run_dir, "metadata.toml")
    shutil.copyfile(priors_path, Path(model_run_dir, "priors.py"))

    if metadata_file.exists():
        with open(metadata_file, "rb") as file:
            metadata = tomllib.load(file)
    else:
        metadata = {}

    new_metadata = {
        "priors_path": str(priors_path),
    }

    metadata.update(new_metadata)

    with open(metadata_file, "wb") as file:
        tomli_w.dump(metadata, file)


def get_available_reports(
    data_dir: str | Path, glob_pattern: str = "*.parquet"
):
    return [
        dt.datetime.strptime(f.stem, "%Y-%m-%d").date()
        for f in Path(data_dir).glob(glob_pattern)
    ]


def main(
    disease: str,
    loc: str,
    report_date: str,
    last_training_date,
    first_training_date,
    model_run_dir,
    facility_level_nssp_data_dir: Path | str,
    state_level_nssp_data_dir: Path | str,
    nwss_data_dir: Path | str,
    param_data_dir: Path | str,
    priors_path: Path | str,
    credentials_path: Path = None,
    nhsn_data_path: Path | str = None,
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

    logger.info(
        f"last training date: {last_training_date}, "
        f"first training date {first_training_date}"
    )
    available_facility_level_reports = get_available_reports(
        facility_level_nssp_data_dir
    )
    available_loc_level_reports = get_available_reports(
        state_level_nssp_data_dir
    )

    facility_level_nssp_data, loc_level_nssp_data = None, None

    if report_date in available_facility_level_reports:
        logger.info("Facility level data available for the given report date")
        facility_datafile = f"{report_date}.parquet"
        facility_level_nssp_data = pl.scan_parquet(
            Path(facility_level_nssp_data_dir, facility_datafile)
        )
    if report_date in available_loc_level_reports:
        logger.info("location-level data available for the given report date.")
        loc_datafile = f"{report_date}.parquet"
        loc_level_nssp_data = pl.scan_parquet(
            Path(state_level_nssp_data_dir, loc_datafile)
        )
    if facility_level_nssp_data is None and loc_level_nssp_data is None:
        raise ValueError(
            f"No data available for the requested report date {report_date}"
        )

    available_nwss_reports = get_available_nwss_reports(nwss_data_dir, disease)
    if report_date in available_nwss_reports:
        nwss_data_raw = pl.scan_parquet(
            Path(
                nwss_data_dir,
                f"NWSS-ETL-{nwss_data_disease_map[disease]}-{report_date}",
                "bronze.parquet",
            )
        )
        nwss_data_cleaned = (
            clean_nwss_data(nwss_data_raw)
            .filter(
                (pl.col("location") == loc)
                & (pl.col("date") >= first_training_date)
            )
            .collect()
        )
        loc_level_nwss_data = preprocess_ww_data(nwss_data_cleaned)
    else:
        raise ValueError(
            "NWSS data not available for the requested report date "
            f"{report_date}"
        )

    logger.info(f"Processing {loc}")
    process_and_save_loc_data(
        loc_abb=loc,
        disease=disease,
        report_date=report_date,
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        model_run_dir=model_run_dir,
        logger=logger,
        facility_level_nssp_data=facility_level_nssp_data,
        loc_level_nssp_data=loc_level_nssp_data,
        loc_level_nwss_data=loc_level_nwss_data,
        credentials_dict=credentials_dict,
        nhsn_data_path=nhsn_data_path,
    )

    logger.info("Generating epiweekly datasets from daily datasets...")
    generate_epiweekly_data(model_run_dir)

    logger.info(f"Preparing model parameeters for {loc}")

    process_and_save_loc_param(
        loc_abb=loc,
        disease=disease,
        loc_level_nwss_data=loc_level_nwss_data,
        param_data_dir=param_data_dir,
        model_run_dir=model_run_dir,
    )

    logger.info(f"Copying and recording priors from {priors_path}...")
    copy_and_record_priors(priors_path, model_run_dir)

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
        default=dt.datetime.today().strftime("%Y-%m-%d"),
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
        "--priors-path",
        type=Path,
        help=(
            "Path to an executible python file defining random variables "
            "that require priors as pyrenew RandomVariable objects."
        ),
        required=True,
    )

    parser.add_argument(
        "--n-training-days",
        type=int,
        default=180,
        help="Number of training days (default: 180).",
    )

    parser.add_argument(
        "--nhsn-data-path",
        type=Path,
        help=("Path to local NHSN data (for local testing)"),
        default=None,
    )

    parser.add_argument(
        "--param-data-dir",
        type=Path,
        default=Path("private_data", "prod_param_estimates"),
        help=(
            "Directory in which to look for parameter estimates"
            "such as delay PMFs."
        ),
        required=True,
    )

    args = parser.parse_args()
    main(**vars(args))
