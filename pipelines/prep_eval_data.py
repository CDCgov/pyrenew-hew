import datetime as dt
import logging
from pathlib import Path

import polars as pl

from pipelines.prep_data import (
    combine_surveillance_data,
    get_loc_pop_df,
    get_nhsn,
    process_loc_level_data,
)


def save_eval_data(
    loc: str,
    disease: str,
    first_training_date,
    last_training_date,
    latest_comprehensive_path: Path | str,
    output_data_dir: Path | str,
    last_eval_date: dt.date = None,
    output_file_name: str = "eval_data.tsv",
    credentials_dict: dict = None,
    nhsn_data_path: Path | str = None,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Reading in truth data...")
    loc_level_nssp_data = pl.scan_parquet(latest_comprehensive_path)

    if last_eval_date is not None:
        loc_level_nssp_data = loc_level_nssp_data.filter(
            pl.col("reference_date") <= last_eval_date
        )

    nssp_data = (
        process_loc_level_data(
            loc_level_nssp_data=loc_level_nssp_data,
            loc_abb=loc,
            disease=disease,
            first_training_date=first_training_date,
            loc_pop_df=get_loc_pop_df(),
        )
        .with_columns(data_type=pl.lit("eval"))
        .pivot(
            on="disease",
            values="ed_visits",
        )
        .rename({disease: "observed_ed_visits"})
        .with_columns(other_ed_visits=pl.col("Total") - pl.col("observed_ed_visits"))
        .drop(pl.col("Total"))
        .sort("date")
    )

    nhsn_data = (
        get_nhsn(
            start_date=first_training_date,
            end_date=None,
            disease=disease,
            loc_abb=loc,
            credentials_dict=credentials_dict,
            local_data_file=nhsn_data_path,
        )
        .filter(
            pl.col("weekendingdate") >= first_training_date
        )  # in testing mode, this isn't guaranteed
        .with_columns(data_type=pl.lit("eval"))
    )

    combined_eval_dat = combine_surveillance_data(
        nssp_data=nssp_data,
        nhsn_data=nhsn_data,
        disease=disease,
    )

    combined_eval_dat.write_csv(
        Path(output_data_dir, "combined_" + output_file_name), separator="\t"
    )
    return None
