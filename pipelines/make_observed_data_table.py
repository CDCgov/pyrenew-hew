import argparse
import datetime
from pathlib import Path

import polars as pl
from prep_data import aggregate_facility_level_nssp_to_state, get_state_pop_df


def save_observed_data_table(
    path_to_latest_data: str | Path, output_path: str | Path
):
    data = pl.scan_parquet(path_to_latest_data)

    state_pop = get_state_pop_df()

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

    full_table = (
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

    full_table.write_csv(output_path, separator="\t")


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
    "output_path", type=Path, help="Save the output tsv file to this path."
)

if __name__ == "__main__":
    args = parser.parse_args()
    save_observed_data_table(**vars(args))
