import argparse
import logging
import datetime
from pathlib import Path

import polars as pl


def main(
        nssp_data_dir,
        output_path,
        report_date: str | datetime.date,
        first_date_to_pull: str | datetime.date = None,
        separator="\t",
        diseases=["COVID-19/Omicron",
                  "Influenza",
                  "RSV"],
):
    diseases_to_pull = diseases + ["Total"]

    if isinstance(report_date, str):
        if report_date == "latest":
            report_date = max(
                f.stem for f in Path(nssp_data_dir).glob("*.parquet"))
        else:
            report_date = datetime.datetime.strptime(
                report_date, "%Y-%m-%d").date()
    elif not isinstance(report_date, datetime.date):
        raise ValueError(
            "`report_date` must be either be a "
            "a `datetime.date` object, or a string "
            "giving a date in IS08601 format.")

    if first_date_to_pull is None:
        first_date_to_pull = pl.col("date").min()
    elif isinstance(first_date_to_pull, str):
        first_date_to_pull = datetime.datetime.strptime(
            first_date_to_pull, "%Y-%m-%d").date()
    elif not isinstance(first_date_to_pull, datetime.date):
        raise ValueError(
            "`first_date_to_pull` must be `None` "
            "in which case all available dates are pulled, ",
            "a `datetime.date` object, or a string "
            "giving a date in IS08601 format.")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Report date: {report_date}")

    datafile = f"{report_date}.parquet"
    nssp_data = pl.scan_parquet(Path(nssp_data_dir, datafile))

    data = nssp_data.filter(
            pl.col("disease").is_in(diseases_to_pull),
            pl.col("metric") == "count_ed_visits",
            pl.col("reference_date") > first_date_to_pull,
            pl.col("report_date") == report_date
        ).select(
            ["reference_date",
             "geo_value",
             "disease",
             "value"]
        ).group_by(
            ["reference_date",
             "geo_value",
             "disease"]
        ).agg(
            value=pl.col("value").sum()
        ).unpivot(
            pl.col("disease")
        ).sort(
            ["reference_date", "geo_value"]
        ).collect()

    logger.info(f"Saving data to {output_path}.")

    data.write_csv(data, separator=separator)

    logger.info("Data preparation complete.")


parser = argparse.ArgumentParser(
    description="Pull NSSP data across pathogens."
)
parser.add_argument(
    "--nssp-data-dir",
    type=Path,
    required=True,
    help="Directory in which to look for NSSP input data."
)
parser.add_argument(
    "--output-path",
    type=Path,
    required=True,
    help="Path to which to save the file."
)

parser.add_argument(
    "--report-date",
    type=str,
    default="latest",
    help="Report date in YYYY-MM-DD format or latest (default: latest)",
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
