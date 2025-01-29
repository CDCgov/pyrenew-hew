import polars as pl
from datetime import timedelta

from preprocess import preprocess_ww_data, indicate_ww_exclusions, preprocess_count_data


def get_input_ww_data(
    ww_data_path,
    forecast_date_i,
    location,
    last_hosp_data_date,
    calibration_time: float,
    for_eval: bool = False,
) -> pl.DataFrame:
    schema_overrides = {
        "county_names": pl.Utf8,
        "major_lab_method": pl.Utf8,
    }
    raw_nwss_data = pl.read_csv(
        ww_data_path,
        schema_overrides=schema_overrides,
    )
    all_ww_data = clean_and_filter_nwss_data(raw_nwss_data)
    ww_data = clean_ww_data(all_ww_data)
    subsetted_ww_data = (
        ww_data.filter(
            (pl.col("location").is_in([location]))
            & (
                pl.col("date")
                >= (
                    last_hosp_data_date
                    - timedelta(days=calibration_time)
                    + timedelta(days=1)
                )
            )
        )
        .with_columns(
            pl.col(["site_pop"])
            .mean()
            .over(["lab", "site", "date", "location"])
            .cast(pl.Int64)
        )
        .with_columns(
            pl.col(["log_genome_copies_per_ml", "log_lod"])
            .mean()
            .over(["lab", "site", "date", "location"])
        )  # why is this necessary?
    )
    ww_data_preprocessed = preprocess_ww_data(
        subsetted_ww_data
    )  # calls validate_ww_conc_data and flag_ww_outliers

    if for_eval:
        ww_data_to_fit = ww_data_preprocessed.with_columns(
            pl.col("log_genome_copies_per_ml").exp().alias("ww")
        ).rename({"below_lod": "below_LOD"})
    else:
        ww_data_to_fit = indicate_ww_exclusions(
            ww_data_preprocessed,
            outlier_col_name="flag_as_ww_outlier",
            remove_outliers=True,
        ).with_columns(pl.lit(forecast_date_i).cast(pl.Date).alias("forecast_date"))
    return ww_data_to_fit


def get_input_hosp_data(
    hosp_data_path: str,
    forecast_date_i: str,
    location_i: str,
    calibration_time: float,
    for_eval: bool = False,
) -> pl.DataFrame:
    """
    Get input hospital admissions data.

    Parameters:
    ----------
    hosp_data_path : str
        A string indicating the path to the directory containing
        time-stamped hospital admissions datasets.
    forecast_date_i : str
        The forecast date for this iteration.
    location_i :
        The location (state) for this iteration.
    calibration_time : float
        A numeric indicating the number of days of model calibration
    for_eval : bool, optional
        A boolean indicating whether the hospital admissions data
        is for evaluation. Default is False, which means it will be used
        to fit a single model. True means we will combine with multiple
        locations and a longer time span than we would fit to.

    Returns:
    -------
    pl.DataFrame
        A Polars DataFrame containing the preprocessed hospital admissions data ready
        to be passed into the wwinference function.
    """
    schema = {
        "daily_hosp_admits": pl.Int32,
        "pop": pl.Int32,
    }
    hosp = pl.read_csv(hosp_data_path, schema_overrides=schema, null_values=["NA"])
    input_hosp = (
        hosp.rename({"ABBR": "location"})
        .with_columns(pl.col("date").str.to_date(format="%Y-%m-%d"))
        .filter(
            (pl.col("location").is_in([location_i]))
            & (
                pl.col("date")
                >= (
                    pl.col("date").max()
                    - timedelta(days=calibration_time)
                    + timedelta(days=1)
                )
            )
        )
    )

    if for_eval:
        hosp_data_preprocessed = input_hosp
    else:
        hosp_data_preprocessed = preprocess_count_data(
            input_hosp, count_col_name="daily_hosp_admits", pop_size_col_name="pop"
        ).with_columns(
            [
                pl.lit(forecast_date_i)
                .alias("forecast_date")
                .str.to_date(format="%Y-%m-%d"),
                pl.lit(location_i).alias("location"),
            ]
        )
    return hosp_data_preprocessed


def get_last_hosp_data_date(input_hosp: pl.DataFrame):
    last_hosp_data_date = input_hosp["date"].max()
    return last_hosp_data_date


def clean_and_filter_nwss_data(raw_nwss_data):
    nwss_subset_raw = (
        raw_nwss_data.filter(
            pl.col("sample_location") == "wwtp",
            pl.col("sample_matrix") != "primary sludge",
            pl.col("pcr_target_units") != "copies/g dry sludge",
            pl.col("pcr_target") == "sars-cov-2",
            pl.col("lab_id").is_not_null(),
            pl.col("wwtp_name").is_not_null(),
        )
        .select(
            [
                "lab_id",
                "sample_collect_date",
                "wwtp_name",
                "pcr_target_avg_conc",
                "wwtp_jurisdiction",
                "population_served",
                "pcr_target_units",
                "pcr_target_below_lod",
                "lod_sewage",
                "quality_flag",
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("pcr_target_units") == "copies/l wastewater")
                .then(pl.col("pcr_target_avg_conc") / 1000)
                .when(pl.col("pcr_target_units") == "log10 copies/l wastewater")
                .then((10 ** pl.col("pcr_target_avg_conc")) / 1000),
                pl.when(pl.col("pcr_target_units") == "copies/l wastewater")
                .then(pl.col("lod_sewage") / 1000)
                .when(pl.col("pcr_target_units") == "log10 copies/l wastewater")
                .then((10 ** pl.col("lod_sewage")) / 1000),
            ]
        )
        .filter(
            (
                ~pl.col("quality_flag").is_in(
                    [
                        "yes",
                        "y",
                        "result is not quantifiable",
                        "temperature not assessed upon arrival at the laboratory",
                        "> max temp and/or hold time",
                    ]
                )
            )
            | (pl.col("quality_flag").is_null())
        )
    )
    conservative_lod = (
        nwss_subset_raw.select(pl.col("lod_sewage").drop_nulls())
        .quantile(0.95)
        .to_numpy()
        .item()
    )
    nwss_subset = nwss_subset_raw.with_columns(
        [
            pl.when(pl.col("lod_sewage").is_null())
            .then(conservative_lod)
            .otherwise(pl.col("lod_sewage"))
            .alias("lod_sewage"),
            pl.col("sample_collect_date")
            .str.to_date(format="%Y-%m-%d")
            .alias("sample_collect_date"),
        ]
    )
    nwss_subset_clean = (
        nwss_subset.group_by(["wwtp_name", "lab_id", "sample_collect_date"])
        .agg([pl.col("pcr_target_avg_conc").mean().alias("pcr_target_avg_conc")])
        .join(
            nwss_subset,
            on=["wwtp_name", "lab_id", "sample_collect_date"],
            how="left",
        )
        .select(
            [
                "sample_collect_date",
                "wwtp_name",
                "lab_id",
                "pcr_target_avg_conc",
                "wwtp_jurisdiction",
                "lod_sewage",
                "population_served",
            ]
        )
    )
    nwss_subset_clean_pop = (
        nwss_subset_clean.group_by("wwtp_name")
        .agg(
            [
                pl.col("population_served")
                .mean()
                .round()
                .cast(pl.Int64)
                .alias("population_served")
            ]
        )
        .join(nwss_subset_clean, on=["wwtp_name"], how="left")
        .select(
            [
                "sample_collect_date",
                "wwtp_name",
                "lab_id",
                "pcr_target_avg_conc",
                "wwtp_jurisdiction",
                "lod_sewage",
                "population_served",
            ]
        )
        .unique(["wwtp_name", "lab_id", "sample_collect_date"])
    )
    return nwss_subset_clean_pop


def clean_ww_data(nwss_subset_clean_pop, log_offset: float = 1e-20):
    """
    Parameters
    ----------
    nwss_subset:
        The raw nwss data filtered down to only the columns we use
    log_offset:  float
        a small numeric value to prevent numerical instability in
    converting from natural scale to log scale

    Return
    ------
    A site-lab level dataset with names and variables that can be used
    for model fitting
    """
    ww_data = (
        nwss_subset_clean_pop.rename(
            {
                "sample_collect_date": "date",
                "population_served": "site_pop",
                "wwtp_jurisdiction": "location",
                "wwtp_name": "site",
                "lab_id": "lab",
                "pcr_target_avg_conc": "log_genome_copies_per_ml",
                "lod_sewage": "log_lod",
            }
        )
        .with_columns(
            [
                (pl.col("log_genome_copies_per_ml") + log_offset)
                .log()
                .alias("log_genome_copies_per_ml"),
                pl.col("log_lod").log().alias("log_lod"),
                pl.col("location").str.to_uppercase().alias("location"),
            ]
        )
        .select(
            [
                "date",
                "site",
                "lab",
                "log_genome_copies_per_ml",
                "log_lod",
                "site_pop",
                "location",
            ]
        )
    )
    return ww_data
