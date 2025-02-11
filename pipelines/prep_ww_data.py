import datetime

import polars as pl


def clean_and_filter_nwss_data(nwss_data):
    """
    Parameters
    ----------
    nwss_data:
        vintaged/pulled nwss data

    Return
    ------
    A site-lab level dataset, filtered to only the columns we use
    for model fitting
    """
    nwss_subset = (
        nwss_data.filter(
            pl.col("sample_location") == "wwtp",
            pl.col("sample_matrix") != "primary sludge",
            pl.col("pcr_target_units") != "copies/g dry sludge",
            pl.col("pcr_target") == "sars-cov-2",
            pl.col("lab_id").is_not_null(),
            pl.col("wwtp_name").is_not_null(),
            pl.col("lod_sewage").is_not_null(),
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
                "lod_sewage",
                "quality_flag",
            ]
        )
        .with_columns(
            pcr_target_avg_conc=pl.when(
                pl.col("pcr_target_units") == "copies/l wastewater"
            )
            .then(pl.col("pcr_target_avg_conc") / 1000)
            .when(pl.col("pcr_target_units") == "log10 copies/l wastewater")
            .then((10 ** pl.col("pcr_target_avg_conc")) / 1000)
            .otherwise(None),
            lod_sewage=pl.when(
                pl.col("pcr_target_units") == "copies/l wastewater"
            )
            .then(pl.col("lod_sewage") / 1000)
            .when(pl.col("pcr_target_units") == "log10 copies/l wastewater")
            .then((10 ** pl.col("lod_sewage")) / 1000)
            .otherwise(None),
            sample_collect_date=pl.col("sample_collect_date").str.to_date(
                format="%Y-%m-%d"
            ),
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
    ).drop(["quality_flag", "pcr_target_units"])

    # Remove if any exact duplicates of pcr_target_avg_conc
    # values present for each combination of wwtp_name, lab_id,
    # and sample_collect_date
    nwss_subset_clean = nwss_subset.unique(
        subset=[
            "sample_collect_date",
            "wwtp_name",
            "lab_id",
            "pcr_target_avg_conc",
        ]
    )

    # replaces time-varying population if present in the NWSS dataset.
    # Model does not allow time varying population
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
        .unique(
            [
                "wwtp_name",
                "lab_id",
                "sample_collect_date",
                "pcr_target_avg_conc",
            ]
        )
    )

    ww_data = (
        nwss_subset_clean_pop.rename(
            {
                "sample_collect_date": "date",
                "population_served": "site_pop",
                "wwtp_jurisdiction": "location",
                "wwtp_name": "site",
                "lab_id": "lab",
            }
        )
        .with_columns(
            [
                pl.col("pcr_target_avg_conc")
                .log()
                .alias("log_genome_copies_per_ml"),
                pl.col("lod_sewage").log().alias("log_lod"),
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


def check_missing_values(df: pl.DataFrame, columns: list[str]):
    """Raises an error if missing values in a given column(s)."""
    missing_cols = [col for col in columns if df[col].has_nulls()]
    if missing_cols:
        raise ValueError(f"Missing values in column(s): {missing_cols}")


def check_column_type(
    df: pl.DataFrame, col_name: str, expected_types: list[type]
):
    """Raises an error if a column's dtype is not in the expected types."""
    if df[col_name].dtype not in expected_types:
        raise TypeError(
            f"{col_name} expected to be one of the following type "
            "{expected_types}, provided type is {df[col_name].dtype}"
        )


def validate_ww_conc_data(
    ww_data: pl.DataFrame,
    conc_col_name: str = "log_genome_copies_per_ml",
    lod_col_name: str = "log_lod",
    date_col_name: str = "date",
    wwtp_col_name: str = "site",
    wwtp_pop_name: str = "site_pop",
    lab_col_name: str = "lab",
):
    """
    Checks nwss data for missing values and data types.
    """
    if ww_data.is_empty():
        raise ValueError("Input DataFrame 'ww_data' is empty.")

    check_missing_values(
        ww_data,
        [
            conc_col_name,
            lod_col_name,
            date_col_name,
            wwtp_col_name,
            wwtp_pop_name,
            lab_col_name,
        ],
    )

    check_column_type(
        ww_data,
        conc_col_name,
        [
            pl.Float32,
            pl.Float64,
        ],
    )
    check_column_type(
        ww_data,
        lod_col_name,
        [
            pl.Float32,
            pl.Float64,
        ],
    )
    check_column_type(ww_data, date_col_name, [pl.Date])
    check_column_type(ww_data, wwtp_col_name, [pl.Int32, pl.Int64, pl.Utf8])
    check_column_type(ww_data, lab_col_name, [pl.Int32, pl.Int64, pl.Utf8])
    check_column_type(ww_data, wwtp_pop_name, [pl.Int32, pl.Int64])

    if (ww_data[wwtp_pop_name] < 0).any():
        raise ValueError("Site populations have negative values.")

    if (
        not ww_data.group_by(wwtp_col_name)
        .n_unique()
        .get_column(wwtp_pop_name)
        .eq(1)
        .all()
    ):
        raise ValueError(
            "The data contains sites with varying population sizes."
        )

    return None


def preprocess_ww_data(
    ww_data,
    conc_col_name: str = "log_genome_copies_per_ml",
    lod_col_name: str = "log_lod",
    date_col_name: str = "date",
    wwtp_col_name: str = "site",
    wwtp_pop_name: str = "site_pop",
    lab_col_name: str = "lab",
):
    """
    Creates indices for wastewater-treatment plant names and
    flag concentration data below the level of detection.

    """
    assert (
        conc_col_name in ww_data.columns
    ), f"Column '{conc_col_name}' is missing in the input data."
    assert (
        lod_col_name in ww_data.columns
    ), f"Column '{lod_col_name}' is missing in the input data."

    validate_ww_conc_data(
        ww_data,
        conc_col_name=conc_col_name,
        lod_col_name=lod_col_name,
        date_col_name=date_col_name,
    )
    ww_data_ordered = ww_data.sort(by=wwtp_pop_name, descending=True)
    lab_site_df = (
        ww_data_ordered.select([lab_col_name, wwtp_col_name])
        .unique()
        .with_columns(pl.arange(0, pl.len()).alias("lab_site_index"))
    )
    site_df = (
        ww_data_ordered.select([wwtp_col_name])
        .unique()
        .with_columns(pl.arange(0, pl.len()).alias("site_index"))
    )
    ww_preprocessed = (
        ww_data_ordered.join(
            lab_site_df, on=[lab_col_name, wwtp_col_name], how="left"
        )
        .join(site_df, on=wwtp_col_name, how="left")
        .rename(
            {
                lod_col_name: "log_lod",
                conc_col_name: "log_genome_copies_per_ml",
            }
        )
        .with_columns(
            [
                (
                    "Site: "
                    + pl.col(wwtp_col_name).cast(pl.Utf8)
                    + ", Lab: "
                    + pl.col(lab_col_name).cast(pl.Utf8)
                ).alias("lab_site_name"),
                (pl.col("log_genome_copies_per_ml") <= pl.col("log_lod"))
                .cast(pl.Int8)
                .alias("below_lod"),
            ]
        )
    )
    return ww_preprocessed


def get_nwss_data(
    ww_data_path,
    start_date: datetime.date,
    state_abb: str,
) -> pl.DataFrame:
    schema_overrides = {
        "county_names": pl.Utf8,
        "major_lab_method": pl.Utf8,
    }
    nwss_data = pl.read_csv(
        ww_data_path,
        schema_overrides=schema_overrides,
    )  # placeholder: TBD: If using a direct API call to decipher or ABS vintage
    ww_data = clean_and_filter_nwss_data(nwss_data).filter(
        (pl.col("location").is_in([state_abb]))
        & (pl.col("date") >= start_date)
    )

    return ww_data
