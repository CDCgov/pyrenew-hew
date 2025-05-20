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
import numpyro.distributions as dist
import polars as pl
import polars.selectors as cs
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from scipy.optimize import minimize

_disease_map = {
    "COVID-19": "COVID-19/Omicron",
}

_inverse_disease_map = {v: k for k, v in _disease_map.items()}


def get_nhsn(
    start_date: datetime.date,
    end_date: datetime.date,
    disease: str,
    loc_abb: str,
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

    loc_abb_for_query = loc_abb if loc_abb != "US" else "USA"

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
            jurisdictions = {py_scalar_to_r_scalar(loc_abb_for_query)}
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
        raise RuntimeError(
            f"pull_and_save_nhsn: {result.stderr.decode('utf-8')}"
        )
    raw_dat = pl.read_parquet(temp_file)
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


def process_loc_level_data(
    loc_level_nssp_data: pl.LazyFrame,
    loc_abb: str,
    disease: str,
    first_training_date: datetime.date,
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
            loc_pop_df.filter(pl.col("abb") != "US").get_column("abb").unique()
        )
        logger.info("Aggregating loc-level data to national")
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
            pl.col("geo_type") == "loc",
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
        .collect(streaming=True)
        # setting streaming = True explicitly
        # avoids an `Option::unwrap()` on a `None` value
        # error. Cause of error not known but presumably
        # related to how parquets are processed.
    )


def get_loc_pop_df():
    return forecasttools.location_table.select(
        pl.col("short_name").alias("abb"),
        pl.col("long_name").alias("name"),
        pl.col("population"),
    )


def get_pmfs(param_estimates: pl.LazyFrame, loc_abb: str, disease: str):
    generation_interval_pmf = (
        param_estimates.filter(
            (pl.col("geo_value").is_null())
            & (pl.col("disease") == disease)
            & (pl.col("parameter") == "generation_interval")
            & (pl.col("end_date").is_null())  # most recent estimate
        )
        .collect(streaming=True)
        .get_column("value")
        .item(0)
        .to_list()
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
        .item(0)
        .to_list()
    )

    # ensure 0 first entry; we do not model the possibility
    # of a zero infection-to-recorded-admission delay in Pyrenew-HEW
    delay_pmf[0] = 0.0
    delay_pmf = jnp.array(delay_pmf)
    delay_pmf = delay_pmf / delay_pmf.sum()
    delay_pmf = delay_pmf.tolist()

    right_truncation_pmf = (
        param_estimates.filter(
            (pl.col("geo_value") == loc_abb)
            & (pl.col("disease") == disease)
            & (pl.col("parameter") == "right_truncation")
            & (pl.col("end_date").is_null())
        )
        .filter(pl.col("reference_date") == pl.col("reference_date").max())
        .collect(streaming=True)
        .get_column("value")
        .item(0)
        .to_list()
    )

    return (generation_interval_pmf, delay_pmf, right_truncation_pmf)


def approx_lognorm(
    pmf: ArrayLike, loc_guess, scale_guess, method: str = "Nelder-Mead"
) -> tuple[float, float]:
    """
    Find loc and scale parameters
    of a lognormal distribution such that
    the lognormal PDF is approxmimately
    proportional to the given discrete PMF.

    Parameters
    ----------
    pmf
       Array representing the PMF.

    loc_guess
       Initial loc value to pass to the optimizer.

    scale_guess
       Initial scale value to pass to the optimizer.

    method
       Optimization method. Passed as the ``method``
       keyword argument to :func:`scipy.optimize.minimize`.
       Default ``"Nelder-Mead"``.

    Returns
    -------
    tuple[float, float]
       A tuple containing the loc parameter as the first
       entry and the scale parameter as the second.

    Raises
    ------
    ValueError
       If optimization fails.
    """
    log_pmf = jnp.log(pmf)
    n = log_pmf.size

    def err(loc_and_scale):
        """
        Our objective function: the squared
        errors of log prob values
        """
        lnorm = dist.LogNormal(loc=loc_and_scale[0], scale=loc_and_scale[1])
        lp = lnorm.log_prob(jnp.arange(1, n + 1))
        normed_lp = lp - logsumexp(lp)
        return jnp.sum((log_pmf - normed_lp) ** 2)

    result = minimize(
        err, jnp.array([loc_guess, scale_guess]), method="Nelder-Mead"
    )
    if not result.success:
        print(result)
        raise ValueError("Discretized lognormal approximation to PMF failed")
    else:
        res = result.x
        return (float(res[0]), float(res[1]))


def process_and_save_loc(
    loc_abb: str,
    disease: str,
    report_date: datetime.date,
    first_training_date: datetime.date,
    last_training_date: datetime.date,
    param_estimates: pl.LazyFrame,
    model_run_dir: Path,
    logger: Logger = None,
    facility_level_nssp_data: pl.LazyFrame = None,
    loc_level_nssp_data: pl.LazyFrame = None,
    loc_level_nwss_data: pl.LazyFrame = None,
    credentials_dict: dict = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if facility_level_nssp_data is None and loc_level_nssp_data is None:
        raise ValueError(
            "Must provide at least one "
            "of facility-level and loc-level"
            "NSSP data"
        )

    loc_pop_df = get_loc_pop_df()

    loc_pop = loc_pop_df.filter(pl.col("abb") == loc_abb).item(0, "population")

    (generation_interval_pmf, delay_pmf, right_truncation_pmf) = get_pmfs(
        param_estimates=param_estimates, loc_abb=loc_abb, disease=disease
    )

    inf_to_hosp_admit_lognormal_loc, inf_to_hosp_admit_lognormal_scale = (
        approx_lognorm(
            jnp.array(delay_pmf)[1:],  # only fit the non-zero delays
            loc_guess=0,
            scale_guess=0.5,
        )
    )

    right_truncation_offset = (report_date - last_training_date).days

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

    nhsn_training_data = get_nhsn(
        start_date=first_training_date,
        end_date=last_training_date,
        disease=disease,
        loc_abb=loc_abb,
        credentials_dict=credentials_dict,
    ).with_columns(pl.lit("train").alias("data_type"))

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
            pop_fraction = subpop_sizes / loc_pop

    data_for_model_fit = {
        "inf_to_hosp_admit_pmf": delay_pmf,
        "inf_to_hosp_admit_lognormal_loc": inf_to_hosp_admit_lognormal_loc,
        "inf_to_hosp_admit_lognormal_scale": inf_to_hosp_admit_lognormal_scale,
        "generation_interval_pmf": generation_interval_pmf,
        "right_truncation_pmf": right_truncation_pmf,
        "loc_pop": loc_pop,
        "right_truncation_offset": right_truncation_offset,
        "nwss_training_data": nwss_training_data,
        "nssp_training_data": nssp_training_data.to_dict(as_series=False),
        "nhsn_training_data": nhsn_training_data.to_dict(as_series=False),
        "nhsn_step_size": nhsn_step_size,
        "pop_fraction": pop_fraction.tolist(),
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
