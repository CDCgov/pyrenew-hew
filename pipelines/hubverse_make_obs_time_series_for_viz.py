import argparse
import datetime as dt
import logging
from pathlib import Path

import polars as pl


def main(
    nssp_data_dir,
    output_path,
    report_date: str | dt.date,
    first_date_to_pull: str | dt.date = None,
    separator="\t",
    diseases=["covid", "influenza", "rsv"],
):
    diseases_to_column_names = dict(
        covid="COVID-19/Omicron",
        influenza="Influenza",
        rsv="RSV",
        total="Total",
    )

    diseases_to_pull = [
        diseases_to_column_names.get(disease) for disease in diseases
    ]

    col_names_to_pull = diseases_to_pull + ["Total"]

    if isinstance(report_date, str):
        if report_date == "latest":
            report_date = max(
                f.stem for f in Path(nssp_data_dir).glob("*.parquet")
            )
        report_date = dt.datetime.strptime(report_date, "%Y-%m-%d").date()
    elif not isinstance(report_date, dt.date):
        raise ValueError(
            "`report_date` must be either be a "
            "a `datetime.date` object, or a string "
            "giving a date in IS08601 format."
        )

    if first_date_to_pull is None:
        first_date_to_pull = pl.col("reference_date").min()
    elif isinstance(first_date_to_pull, str):
        first_date_to_pull = dt.datetime.strptime(
            first_date_to_pull, "%Y-%m-%d"
        ).date()
    elif not isinstance(first_date_to_pull, dt.date):
        raise ValueError(
            "`first_date_to_pull` must be `None` "
            "in which case all available dates are pulled, ",
            "a `datetime.date` object, or a string "
            "giving a date in IS08601 format.",
        )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Report date: {report_date}")

    datafile = f"{report_date}.parquet"
    nssp_data = pl.scan_parquet(Path(nssp_data_dir, datafile))

    data = (
        nssp_data.filter(
            pl.col("disease").is_in(col_names_to_pull),
            pl.col("metric") == "count_ed_visits",
            pl.col("reference_date") > first_date_to_pull,
            pl.col("report_date") == report_date,
        )
        .select(["reference_date", "geo_value", "disease", "value"])
        .group_by(["reference_date", "geo_value", "disease"])
        .agg(value=pl.col("value").sum())
        .sort(["reference_date", "geo_value"])
        .collect()
        .pivot(on="disease", index=["reference_date", "geo_value"])
        .rename(
            {
                v: f"count_{k}"
                for k, v in diseases_to_column_names.items()
                if v in col_names_to_pull
            }
        )
        .with_columns(
            **{
                f"frac_{x}": (pl.col(f"count_{x}") / pl.col("count_total"))
                for x in diseases
            }
        )
        .with_columns(
            **{f"pct_{x}": (100.0 * pl.col(f"frac_{x}")) for x in diseases}
        )
        .select(
            [
                pl.col("reference_date").alias("date"),
                pl.col("geo_value").alias("location"),
            ]
            + [
                item
                for x in diseases
                for item in [f"count_{x}", f"frac_{x}", f"pct_{x}"]
            ]
            + ["count_total"]
        )
    )

    print(data)

    logger.info(f"Saving data to {output_path}.")

    data.write_csv(file=output_path, separator=separator)

    logger.info("Data preparation complete.")


parser = argparse.ArgumentParser(
    description="Pull NSSP data across pathogens."
)
parser.add_argument(
    "nssp_data_dir",
    type=Path,
    help=(
        "Directory in which to look for NSSP data gold table .parquet files."
    ),
)
parser.add_argument(
    "output_path",
    type=Path,
    help="Path to which to save the output file, as a tsv.",
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
