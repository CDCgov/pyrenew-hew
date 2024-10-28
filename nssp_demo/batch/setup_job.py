import argparse

import polars as pl
from azure.batch import models
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job_if_not_exists
from azuretools.task import get_container_settings, get_task_config


def main(job_id, pool_id, container_image) -> None:
    creds = EnvCredentialHandler()
    client = get_batch_service_client(creds)
    job = models.JobAddParameter(
        id=job_id,
        pool_info=models.PoolInformation(pool_id=pool_id),
    )
    create_job_if_not_exists(client, job, verbose=True)

    container_settings = get_container_settings(
        container_image,
        working_directory="containerImageDefault",
        mount_pairs=[
            {
                "source": "nssp-etl",
                "target": "/pyrenew-hew/nssp_demo/nssp-etl",
            },
            {
                "source": "prod-param-estimates",
                "target": "/pyrenew-hew/nssp_demo/params",
            },
            {
                "source": "pyrenew-test-output",
                "target": "/pyrenew-hew/nssp_demo/private_data",
            },
        ],
    )

    base_call = (
        "/bin/bash -c '"
        "python nssp_demo/forecast_state.py "
        "--disease {disease} "
        "--state {state} "
        "--n-training-days 365 "
        "--n-warmup 1000 "
        "--n-samples 500 "
        "--nssp-data-dir nssp_demo/nssp-etl/gold "
        "--param-data-dir nssp_demo/params "
        "--output-data-dir nssp_demo/private_data"
        "'"
    )

    states = pl.read_csv(
        "https://raw.githubusercontent.com/k5cents/usa/"
        "refs/heads/master/data-raw/states.csv"
    )

    excluded_states = ["GU", "MO", "WY", "PR"]

    all_states = (
        states.filter(~pl.col("abb").is_in(excluded_states))
        .get_column("abb")
        .to_list()
    )

    for disease in ["COVID-19", "Influenza"]:
        for state in all_states:
            task = get_task_config(
                f"{job_id}-{state}-{disease}",
                base_call=base_call.format(state=state, disease=disease),
                container_settings=container_settings,
            )
            client.task.add(job_id, task)
            pass
        pass

    return None


parser = argparse.ArgumentParser()

parser.add_argument("job_id")
parser.add_argument("pool_id")
parser.add_argument("container_image")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.job_id, args.pool_id, args.container_image)
