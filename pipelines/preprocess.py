import polars as pl
from validate import validate_ww_conc_data, validate_count_data


def preprocess_count_data(
    count_data: pl.DataFrame,
    count_col_name: str = "daily_hosp_admits",
    pop_size_col_name: str = "state_pop",
) -> pl.DataFrame:
    """
    Pre-processes hospital admissions data, converting column names to match the
    expected format for further analysis.

    Parameters:
    ----------
    count_data : pl.DataFrame
        A Polars DataFrame containing the following columns: date,
        a count column and  a population size column

    count_col_name : str | optional
        The name of the column containing the epidemiological indiator.
        Default is `daily_hosp_admits`.

    pop_size_col_name : str | optional
        The name of the column containing the population size for each record.
        Default is `state_pop`.

    Returns:
    -------
    pl.DataFrame
        A Polars DataFrame with the columns renamed to:
        - "count" for the admissions count column
        - "total_pop" for the population size column
        - "date" for the date column.
    """

    assert (
        count_col_name in count_data.columns
    ), f"Column '{count_col_name}' is missing in the input data."
    assert (
        pop_size_col_name in count_data.columns
    ), f"Column '{pop_size_col_name}' is missing in the input data."

    validate_count_data(
        count_data, count_col_name=count_col_name, pop_size_col_name=pop_size_col_name
    )
    count_data_preprocessed = count_data.rename(
        {count_col_name: "count", pop_size_col_name: "total_pop"}
    )
    return count_data_preprocessed


def preprocess_ww_data(
    ww_data, conc_col_name="log_genome_copies_per_ml", lod_col_name="log_lod"
):
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
        .rename({lod_col_name: "log_lod", conc_col_name: "log_genome_copies_per_ml"})
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
        .filter(pl.col("below_lod") == 0, pl.col("n_data_points") > threshold_n_dps)
        .sort(by="date", descending=True)  # kept for consistency with R but seee notes
        .rename({"log_genome_copies_per_ml": "log_conc"})
        .with_columns(
            pl.col("date", "log_conc")
            .shift()
            .over("lab_site_index")
            .name.prefix("prev_")
        )
        .with_columns(
            [
                (pl.col("log_conc") - pl.col("prev_log_conc")).alias("diff_log_conc"),
                (
                    pl.col("date").cast(pl.Int64) - pl.col("prev_date").cast(pl.Int64)
                ).alias("diff_time"),
            ]
        )
        .with_columns((pl.col("diff_log_conc") / pl.col("diff_time")).alias("rho"))
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
                ((pl.col("rho") - pl.col("mean_rho")) / pl.col("std_rho")).alias(
                    "z_score_rho"
                ),
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
                    & (pl.col("z_score_rho") * pl.col("z_score_rho_t_plus_1") < 0)
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
        .with_columns([pl.lit(0).alias("exclude")])
    )
    ww_w_outliers_flagged = ww_z_scored.select(
        *ww_data.columns, "flag_as_ww_outlier", "exclude"
    )

    return ww_w_outliers_flagged


def indicate_ww_exclusions(
    data, outlier_col_name="flag_as_ww_outlier", remove_outliers=True
):
    if outlier_col_name not in data.columns:
        raise ValueError(
            f"Specified name of the outlier column '{outlier_col_name}' not present in the data"
        )

    if remove_outliers:
        data = data.with_columns(
            pl.when(pl.col(outlier_col_name) == 1)
            .then(1)
            .otherwise(pl.col("exclude"))
            .alias("exclude")
        )

    return data
