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
    nwss_subset_raw = (
        nwss_data.filter(
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
                .when(
                    pl.col("pcr_target_units") == "log10 copies/l wastewater"
                )
                .then((10 ** pl.col("pcr_target_avg_conc")) / 1000),
                pl.when(pl.col("pcr_target_units") == "copies/l wastewater")
                .then(pl.col("lod_sewage") / 1000)
                .when(
                    pl.col("pcr_target_units") == "log10 copies/l wastewater"
                )
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
        .agg(
            [pl.col("pcr_target_avg_conc").mean().alias("pcr_target_avg_conc")]
        )
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


def validate_ww_conc_data(
    ww_data: pl.DataFrame, conc_col_name: str, lod_col_name: str
):
    """
    Checks nwss data for missing values and data types.
    """
    if ww_data.is_empty():
        raise ValueError("Input DataFrame 'ww_data' is empty.")

    ww_conc = ww_data[conc_col_name]
    if ww_conc.is_null().any():
        raise ValueError(
            f"{conc_col_name} has missing values. "
            "Observations below the limit of detection "
            "must indicate a numeric value less than "
            "the limit of detection."
        )

    if not isinstance(ww_conc, pl.Series):
        raise TypeError(
            "Wastewater concentration is expected to be a 1D Series."
        )

    if ww_conc.dtype not in [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Float32,
        pl.Float64,
    ]:
        raise TypeError(
            "Expected numeric values for wastewater concentration."
        )

    if ww_data["date", "site", "lab"].is_duplicated().any():
        raise ValueError(
            "Duplicate observations found for the same site, lab, and date."
        )

    ww_lod = ww_data[lod_col_name]
    if ww_lod.is_null().any():
        raise ValueError(
            "There are missing values in the limit of detection data."
        )

    if not isinstance(ww_lod, pl.Series):
        raise TypeError(
            "Limit of detection data is expected to be a 1D Series."
        )

    ww_obs_dates = ww_data["date"]
    if ww_obs_dates.is_null().any():
        raise ValueError("Date column has missing values.")

    if ww_obs_dates.dtype != pl.Date:
        raise TypeError("Date column has to be of Date type.")

    site_labels = ww_data["site"]
    if site_labels.is_null().any():
        raise TypeError("Site labels are missing.")

    if (
        site_labels.dtype not in [pl.Int32, pl.Int64]
        and site_labels.dtype != pl.Utf8
    ):
        raise TypeError("Site labels not of integer/string type.")

    lab_labels = ww_data["lab"]
    if lab_labels.is_null().any():
        raise TypeError("Lab labels are missing.")

    if (
        lab_labels.dtype not in [pl.Int32, pl.Int64]
        and lab_labels.dtype != pl.Utf8
    ):
        raise TypeError("Lab labels are not of integer/string type.")

    site_pops = ww_data["site_pop"]
    if site_pops.is_null().any():
        raise ValueError("Site populations are missing.")
    if site_pops.dtype not in [pl.Int32, pl.Int64] or (site_pops < 0).any():
        raise ValueError(
            "Site populations are not integers, or have negative values."
        )

    records_per_site_per_pop = (
        ww_data[["site", "site_pop"]].unique().group_by("site").len()
    )
    if (records_per_site_per_pop["len"] != 1).any():
        raise ValueError(
            "The data contains sites with varying population sizes."
        )

    return None


def flag_ww_outliers(
    ww_data,
    conc_col_name="log_genome_copies_per_ml",
    rho_threshold: float = 2,
    log_conc_threshold: float = 3,
    threshold_n_dps: int = 1,
):
    n_dps = (
        ww_data.filter(pl.col("below_lod") == 0)
        .group_by("lab_site_index")
        .agg(pl.len().alias("n_data_points"))
    )
    ww_stats = (
        ww_data.join(n_dps, on="lab_site_index", how="left")
        .filter(
            pl.col("below_lod") == 0, pl.col("n_data_points") > threshold_n_dps
        )
        .sort(
            by="date", descending=True
        )  # kept for consistency with R but seee notes
        .rename({"log_genome_copies_per_ml": "log_conc"})
        .with_columns(
            pl.col("date", "log_conc")
            .shift()
            .over("lab_site_index")
            .name.prefix("prev_")
        )
        .with_columns(
            [
                (pl.col("log_conc") - pl.col("prev_log_conc")).alias(
                    "diff_log_conc"
                ),
                (
                    pl.col("date").cast(pl.Int64)
                    - pl.col("prev_date").cast(pl.Int64)
                ).alias("diff_time"),
            ]
        )
        .with_columns(
            (pl.col("diff_log_conc") / pl.col("diff_time")).alias("rho")
        )
        .select(["date", "lab_site_index", "rho"])
    )
    ww_rho = ww_data.join(ww_stats, on=["lab_site_index", "date"], how="left")
    ww_rho_stats = ww_rho.with_columns(
        [
            pl.col("rho", conc_col_name)
            .mean()
            .over("lab_site_index")
            .name.prefix("mean_"),
            pl.col("rho", conc_col_name)
            .std()
            .over("lab_site_index")
            .name.prefix("std_"),
        ]
    )
    ww_z_scored = (
        ww_rho_stats.with_columns(
            [
                (
                    (pl.col(conc_col_name) - pl.col("mean_" + conc_col_name))
                    / pl.col("std_" + conc_col_name)
                ).alias("z_score_conc"),
                (
                    (pl.col("rho") - pl.col("mean_rho")) / pl.col("std_rho")
                ).alias("z_score_rho"),
            ]
        )
        .sort(by="date", descending=False)
        .with_columns(
            pl.col("z_score_rho")
            .shift(-1)
            .over("lab_site_index")
            .name.suffix("_t_plus_1")
        )
        .with_columns(
            [
                pl.when(abs(pl.col("z_score_conc")) >= log_conc_threshold)
                .then(1)
                .otherwise(0)
                .alias("flagged_for_removal_conc"),
                pl.when(
                    (abs(pl.col("z_score_rho")) >= rho_threshold)
                    & (abs(pl.col("z_score_rho_t_plus_1")) >= rho_threshold)
                    & (
                        pl.col("z_score_rho") * pl.col("z_score_rho_t_plus_1")
                        < 0
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("flagged_for_removal_rho"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("flagged_for_removal_rho") == 1)
                .then(1)
                .when(pl.col("flagged_for_removal_conc") == 1)
                .then(1)
                .otherwise(0)
                .alias("flag_as_ww_outlier")
            ]
        )
    )

    ww_w_outliers_flagged = ww_z_scored.select(
        *ww_data.columns, "flag_as_ww_outlier"
    )

    return ww_w_outliers_flagged


def preprocess_ww_data(
    ww_data, conc_col_name="log_genome_copies_per_ml", lod_col_name="log_lod"
):
    """
    Creates lab-site-index and flag for wastewater
    concentration data being below the level of detection.

    """
    assert (
        conc_col_name in ww_data.columns
    ), f"Column '{conc_col_name}' is missing in the input data."
    assert (
        lod_col_name in ww_data.columns
    ), f"Column '{lod_col_name}' is missing in the input data."

    validate_ww_conc_data(
        ww_data, conc_col_name=conc_col_name, lod_col_name=lod_col_name
    )
    ww_data_ordered = ww_data.sort(by="site_pop", descending=True)
    lab_site_df = (
        ww_data_ordered.select(["lab", "site"])
        .unique()
        .with_columns(pl.arange(0, pl.len()).alias("lab_site_index"))
    )
    site_df = (
        ww_data_ordered.select(["site"])
        .unique()
        .with_columns(pl.arange(0, pl.len()).alias("site_index"))
    )
    ww_data_add_cols = (
        ww_data_ordered.join(lab_site_df, on=["lab", "site"], how="left")
        .join(site_df, on="site", how="left")
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
                    + pl.col("site").cast(pl.Utf8)
                    + ", Lab: "
                    + pl.col("lab").cast(pl.Utf8)
                ).alias("lab_site_name"),
                (pl.col("log_genome_copies_per_ml") <= pl.col("log_lod"))
                .cast(pl.Int8)
                .alias("below_lod"),
            ]
        )
    )
    ww_preprocessed = flag_ww_outliers(
        ww_data_add_cols, conc_col_name="log_genome_copies_per_ml"
    )
    return ww_preprocessed


def get_site_subpop_spine(input_ww_data, total_pop):
    ww_data_present = input_ww_data is not None
    if ww_data_present:
        # Check if auxiliary subpopulation needs to be added
        add_auxiliary_subpop = (
            total_pop
            > input_ww_data.select(
                pl.col("site_pop", "site", "lab", "lab_site_index")
            )
            .unique()
            .sum()
            .to_numpy()
            .flatten()[0]
        )
        site_indices = (
            input_ww_data.select(["site_index", "site", "site_pop"])
            .unique()
            .sort("site_index")
        )
        if add_auxiliary_subpop:
            aux_subpop = pl.DataFrame(
                {
                    "site_index": [None],
                    "site": [None],
                    "site_pop": [
                        total_pop
                        - site_indices.select(pl.col("site_pop"))
                        .sum()
                        .to_numpy()
                        .flatten()[0]
                    ],
                }
            )
        else:
            aux_subpop = pl.DataFrame()

        site_subpop_spine = (
            pl.concat([aux_subpop, site_indices], how="vertical_relaxed")
            .with_columns(
                [
                    pl.col("site_index").cum_count().alias("subpop_index"),
                    pl.when(pl.col("site").is_not_null())
                    .then(
                        pl.col("site").map_elements(
                            lambda x: f"Site: {x}", return_dtype=str
                        )
                    )
                    .otherwise(pl.lit("remainder of population"))
                    .alias("subpop_name"),
                ]
            )
            .rename({"site_pop": "subpop_pop"})
        )
    else:
        site_subpop_spine = pl.DataFrame(
            {
                "site_index": [None],
                "site": [None],
                "subpop_pop": [total_pop],
                "subpop_index": [1],
                "subpop_name": ["total population"],
            }
        )
    return site_subpop_spine


def get_date_time_spine(
    start_date: datetime.date,
    end_date: datetime.date,
):
    date_time_spine = pl.DataFrame(
        {
            "date": pl.date_range(
                start=start_date, end=end_date, interval="1d", eager=True
            )
        }
    )
    date_time_spine = date_time_spine.with_columns(
        pl.arange(0, pl.len()).alias("t")
    )
    return date_time_spine


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
    ww_data = (
        clean_and_filter_nwss_data(nwss_data)
        .filter(
            (pl.col("location").is_in([state_abb]))
            & (pl.col("date") >= start_date)
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
        )
    )

    return ww_data
