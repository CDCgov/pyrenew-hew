import datetime
import logging
from pathlib import Path

import polars as pl

from prep_data import (
    get_state_pop_df,
    process_state_level_data,
)


def save_eval_data(
    state: str,
    disease: str,
    report_date: datetime.date,
    first_training_date,
    last_training_date,
    latest_comprehensive_path: Path | str,
    output_data_dir: Path | str,
    last_eval_date: datetime.date = None,
    output_file_name: str = "eval_data.tsv",
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Reading in truth data...")
    state_level_nssp_data = pl.scan_parquet(latest_comprehensive_path)

    if last_eval_date is not None:
        state_level_nssp_data = state_level_nssp_data.filter(
            pl.col("reference_date") <= last_eval_date
        )

    state_level_data = (
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

    state_level_data.write_csv(
        Path(output_data_dir, output_file_name), separator="\t"
    )

    return None
