import argparse
import datetime
import logging
from pathlib import Path

import epiweeks
import polars as pl

from pipelines.preprocess.prep_data import (
    aggregate_facility_level_nssp_to_state,
    get_state_pop_df,
)


def save_observed_data_tables(
    path_to_latest_data: str | Path,
    output_dir: str | Path,
    daily_filename: str = "daily.tsv",
    epiweekly_filename: str = "epiweekly.tsv",
):
    """
    Save daily and epiweekly tables of
    observed data to the given directory.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Creating observed data table from {path_to_latest_data}...")

    output_path_daily = Path(output_dir, daily_filename)
    output_path_epiweekly = Path(output_dir, epiweekly_filename)

    data = pl.scan_parquet(path_to_latest_data)

    state_pop = get_state_pop_df()

    logger.info("Pulling and concatenating individual locations...")

    visits_by_disease = [
        pl.concat(
            map(
                lambda abb: aggregate_facility_level_nssp_to_state(
                    facility_level_nssp_data=data,
                    state_abb=abb,
                    disease=disease,
                    first_training_date=datetime.date(2023, 1, 1),
                    state_pop_df=state_pop,
                ),
                ["US"] + [x for x in state_pop["abb"]],
            )
        ).filter(pl.col("disease") == disease)
        for disease in ["COVID-19", "Influenza", "RSV", "Total"]
    ]

    full_daily_table = (
        pl.concat(visits_by_disease)
        .pivot(on="disease", values="ed_visits")
        .select(
            date=pl.col("date"),
            location=pl.col("geo_value"),
            count_covid=pl.col("COVID-19"),
            frac_covid=pl.col("COVID-19") / pl.col("Total"),
            pct_covid=100 * pl.col("COVID-19") / pl.col("Total"),
            count_influenza=pl.col("Influenza"),
            frac_influenza=pl.col("Influenza") / pl.col("Total"),
            pct_influenza=100 * pl.col("Influenza") / pl.col("Total"),
            count_rsv=pl.col("RSV"),
            frac_rsv=pl.col("RSV") / pl.col("Total"),
            pct_rsv=100 * pl.col("RSV") / pl.col("Total"),
            count_total=pl.col("Total"),
        )
        .sort(["location", "date"])
    )

    logger.info("Making epiweekly table...")

    full_epiweekly_table = (
        full_daily_table.with_columns(
            epiweek=pl.col("date").map_elements(
                lambda x: epiweeks.Week.fromdate(x).week, return_dtype=pl.Int64
            ),
            epiyear=pl.col("date").map_elements(
                lambda x: epiweeks.Week.fromdate(x).year, return_dtype=pl.Int64
            ),
        )
        .group_by(["location", "epiweek", "epiyear"])
        .agg(
            count_covid=pl.col("count_covid").sum(),
            count_influenza=pl.col("count_influenza").sum(),
            count_rsv=pl.col("count_rsv").sum(),
            count_total=pl.col("count_total").sum(),
        )
        .with_columns(epiweek_and_year=pl.struct(["epiweek", "epiyear"]))
        .with_columns(
            prop_covid=pl.col("count_covid") / pl.col("count_total"),
            prop_influenza=pl.col("count_influenza") / pl.col("count_total"),
            prop_rsv=pl.col("count_rsv") / pl.col("count_total"),
            reference_date=pl.col("epiweek_and_year").map_elements(
                lambda x: epiweeks.Week(
                    week=x["epiweek"], year=x["epiyear"]
                ).enddate(),
                return_dtype=pl.Date,
            ),
        )
        .select(
            [
                "location",
                "epiweek",
                "epiyear",
                "reference_date",
                "count_covid",
                "prop_covid",
                "count_influenza",
                "prop_influenza",
                "count_rsv",
                "prop_rsv",
                "count_total",
            ]
        )
        .sort(["location", "reference_date"])
    )

    logger.info(f"Saving tables to {output_dir}...")

    full_daily_table.write_csv(output_path_daily, separator="\t")
    full_epiweekly_table.write_csv(output_path_epiweekly, separator="\t")

    logger.info("Done.")


parser = argparse.ArgumentParser()

parser.add_argument(
    "path_to_latest_data",
    type=Path,
    help=(
        "Path to a parquet file containing the latest "
        "ED visit observations."
    ),
)

parser.add_argument(
    "output_dir",
    type=Path,
    help="Save the output tsv files to this directory.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    save_observed_data_tables(**vars(args))
