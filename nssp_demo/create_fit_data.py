import os
import pathlib
from datetime import timedelta

import duckdb
import pyarrow.parquet as pq

disease_map = {
    "COVID-19": "COVID-19/Omicron",
    "Influenza": "Influenza",
    "RSV": "RSV",
}
disease = "COVID-19"
# todo read these from cli

# todo also read dates from cli

data_path = pathlib.Path("private_data/2024-10-10.parquet")
nssp_data = duckdb.arrow(pq.read_table(data_path))

all_geos = (
    nssp_data.unique("geo_value")
    .order("geo_value")
    .pl()["geo_value"]
    .to_list()
)

for state_abb in all_geos:
    data_to_save = duckdb.sql(
        f"""
        SELECT report_date, reference_date, SUM(value) AS value
        FROM nssp_data
        WHERE disease = '{disease_map[disease]}' AND metric = 'count_ed_visits'
        AND geo_value = '{state_abb}'
        GROUP BY report_date, reference_date
        ORDER BY report_date, reference_date
        """
    )
    # why not count_admitted_ed_visits ?

    data_to_save_pl = data_to_save.pl()
    report_date = data_to_save_pl[0, "report_date"]
    actual_first_date = data_to_save_pl["reference_date"].min()
    actual_last_date = data_to_save_pl["reference_date"].max()
    last_training_date = actual_last_date - timedelta(days=7)

    model_folder_name = f"{disease.lower()}_r_{report_date}_f_{actual_first_date}_l_{actual_last_date}_t_{last_training_date}"

    model_folder = pathlib.Path("private_data", model_folder_name)
    os.makedirs(model_folder, exist_ok=True)
    data_folder = pathlib.Path(model_folder, state_abb)
    os.makedirs(data_folder, exist_ok=True)
    print(f"Saving {state_abb}")
    data_to_save.to_csv(str(pathlib.Path(data_folder, "data.csv")))
    # todo: save as tsv
