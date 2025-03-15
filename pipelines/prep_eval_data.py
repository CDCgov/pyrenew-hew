import datetime
import logging
from pathlib import Path

import polars as pl
from prep_data import (
    combine_surveillance_data,
    get_nhsn,
    get_state_pop_df,
    process_state_level_data,
)


def save_eval_data(
    state: str,
    disease: str,
    first_training_date,
    last_training_date,
    latest_comprehensive_path: Path | str,
    output_data_dir: Path | str,
    last_eval_date: datetime.date = None,
    output_file_name: str = "eval_data.tsv",
    credentials_dict: dict = None,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Reading in truth data...")
    state_level_nssp_data = pl.scan_parquet(latest_comprehensive_path)

    if last_eval_date is not None:
        state_level_nssp_data = state_level_nssp_data.filter(
            pl.col("reference_date") <= last_eval_date
        )

    nssp_data = (
        process_state_level_data(
            state_level_nssp_data=state_level_nssp_data,
            state_abb=state,
            disease=disease,
            first_training_date=first_training_date,
            state_pop_df=get_state_pop_df(),
        )
        .with_columns(data_type=pl.lit("eval"))
        .sort(["date", "disease"])
    )

    nhsn_data = get_nhsn(
        start_date=first_training_date,
        end_date=None,
        disease=disease,
        state_abb=state,
        credentials_dict=credentials_dict,
    ).with_columns(data_type=pl.lit("eval"))

    combined_eval_dat = combine_surveillance_data(
        nssp_data=nssp_data,
        nhsn_data=nhsn_data,
        disease=disease,
    )

    combined_eval_dat.write_csv(
        Path(output_data_dir, "combined_" + output_file_name), separator="\t"
    )
    return None
