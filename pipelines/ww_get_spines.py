import polars as pl

from datetime import datetime, timedelta


def get_site_subpop_spine(input_ww_data, input_count_data):
    ww_data_present = input_ww_data is not None
    total_pop = (
        input_count_data.select(pl.col("total_pop").unique()).to_numpy().flatten()[0]
    )
    if ww_data_present:
        # Check if auxiliary subpopulation needs to be added
        add_auxiliary_subpop = (
            total_pop
            > input_ww_data.select(pl.col("site_pop"))
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


def get_lab_site_site_spine(input_ww_data):
    ww_data_present = input_ww_data is not None
    if ww_data_present:
        lab_site_site_spine = (
            input_ww_data.select(
                [
                    "lab_site_index",
                    "site_index",
                    "site",
                    "lab",
                    "site_pop",
                    "lab_site_name",
                ]
            )
            .unique()
            .sort("lab_site_index")
        )
    else:
        lab_site_site_spine = pl.DataFrame()
    return lab_site_site_spine


def get_lab_site_subpop_spine(lab_site_site_spine, site_subpop_spine):
    ww_data_present = len(lab_site_site_spine) > 0
    if ww_data_present:
        lab_site_subpop_spine = lab_site_site_spine.join(
            site_subpop_spine,
            on=["site_index", "site"],  # Columns to join on
            how="left",
        )
    else:
        lab_site_subpop_spine = pl.DataFrame({"subpop_index": pl.Int64})
    return lab_site_subpop_spine


def get_date_time_spine(
    forecast_date,
    input_count_data,
    calibration_time,
):
    forecast_date = datetime.strptime(forecast_date, "%Y-%m-%d").date()
    last_count_data_date = input_count_data["date"].min()
    nowcast_time = (forecast_date - last_count_data_date).days
    start_date = input_count_data["date"].min()
    # Calculate the end date for the sequence, R code includes forecast horizon in date spine
    end_date = start_date + timedelta(days=calibration_time + nowcast_time)
    date_time_spine = pl.DataFrame(
        {
            "date": pl.date_range(
                start=start_date, end=end_date, interval="1d", eager=True
            )
        }
    )
    date_time_spine = date_time_spine.with_columns(pl.arange(0, pl.len()).alias("t"))
    return date_time_spine
