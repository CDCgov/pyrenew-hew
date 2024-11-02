"""
Set up a multi-location, multi-date,
potentially multi-disease end to end
retrospective evaluation run for pyrenew-hew
on Azure Batch.
"""

import argparse
import datetime

import polars as pl
from azure.batch import models
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job_if_not_exists
from azuretools.task import get_container_settings, get_task_config


def main(
    job_id: str,
    pool_id: str,
    diseases: str,
    container_image_name: str = "pyrenew-hew",
    container_image_version: str = "latest",
) -> None:
    """
    job_id
        Name for the Batch job.

    pool_id
        Azure Batch pool on which to run the job.

    diseases
        Name(s) of disease(s) to run as part of the job,
        as a whitespace-separated string. Supported
        values are 'COVID-19' and 'Influenza'.

    container_image_name:
        Name of the container to use for the job.
        This container should exist within the Azure
        Container Registry account associated to
        the job. Default 'pyrenew-hew'.
        The container registry account name and enpoint
        will be obtained from local environm variables
        via a :class``azuretools.auth.EnvCredentialHandler`.

    container_image_version
        Version of the container to use. Default 'latest'.

    Returns
    -------
    None
    """
    supported_diseases = ["COVID-19", "Influenza"]

    disease_list = diseases.split()
    for d in disease_list:
        if d not in supported_diseases:
            raise ValueError(
                f"Unsupported disease '{d}'; supported diseases "
                f"are {supported_diseases}"
            )

    creds = EnvCredentialHandler()
    client = get_batch_service_client(creds)
    job = models.JobAddParameter(
        id=job_id,
        pool_info=models.PoolInformation(pool_id=pool_id),
    )
    create_job_if_not_exists(client, job, verbose=True)

    container_image = (
        f"{creds.azure_container_registry_account}."
        f"{creds.azure_container_registry_domain}/"
        f"{container_image_name}:{container_image_version}"
    )
    container_settings = get_container_settings(
        container_image,
        working_directory="containerImageDefault",
        mount_pairs=[
            {
                "source": "nssp-etl",
                "target": "/pyrenew-hew/nssp_demo/nssp-etl",
            },
            {
                "source": "nssp-archival-vintages",
                "target": "/pyrenew-hew/nssp_demo/nssp-archival-vintages",
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
        "--n-training-days 75 "
        "--n-warmup 1000 "
        "--n-samples 500 "
        "--facility-level-nssp-data-dir nssp_demo/nssp-etl/gold "
        "--state-level-nssp-data-dir "
        "nssp_demo/nssp-archival-vintages/gold "
        "--param-data-dir nssp_demo/params "
        "--output-data-dir nssp_demo/private_data "
        "--report-date {report_date:%Y-%m-%d} "
        "--exclude-last-n-days 5 "
        "--score "
        "--eval-data-path "
        "nssp_demo/nssp-archival-vintages/latest_comprehensive.parquet"
        "'"
    )

    locations = pl.read_csv(
        "https://www2.census.gov/geo/docs/reference/state.txt", separator="|"
    )

    excluded_locations = ["AS", "GU", "MO", "MP", "PR", "UM", "VI", "WY"]

    all_locations = (
        locations.filter(~pl.col("STUSAB").is_in(excluded_locations))
        .get_column("STUSAB")
        .to_list()
    )

    report_dates = [
        datetime.date(2023, 10, 11) + datetime.timedelta(weeks=x)
        for x in range(30)
    ]

    for disease in disease_list:
        for report_date in report_dates:
            for state in all_locations:
                task = get_task_config(
                    f"{job_id}-{state}-{disease}-{report_date}",
                    base_call=base_call.format(
                        state=state,
                        disease=disease,
                        report_date=report_date,
                    ),
                    container_settings=container_settings,
                )
                client.task.add(job_id, task)
                pass
            pass
        pass

    return None


parser = argparse.ArgumentParser()

parser.add_argument("job_id", type=str, help="Name for the Azure batch job")
parser.add_argument(
    "pool_id",
    type=str,
    help=("Name of the Azure batch pool on which to run the job"),
)
parser.add_argument(
    "diseases",
    type=str,
    help=(
        "Name(s) of disease(s) to run as part of the job, "
        "as a whitespace-separated string. Supported "
        "values are 'COVID-19' and 'Influenza'."
    ),
)

parser.add_argument(
    "--container-image-name",
    type=str,
    help="Name of the container to use for the job.",
    default="pyrenew-hew",
)

parser.add_argument(
    "--container-image-version",
    type=str,
    help="Version of the container to use for the job.",
    default="latest",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
