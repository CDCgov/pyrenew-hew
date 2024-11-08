"""
Set up a multi-location, multi-date,
potentially multi-disease end to end
retrospective evaluation run for pyrenew-hew
on Azure Batch.
"""

import argparse
import itertools

import polars as pl
from azure.batch import models
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job_if_not_exists
from azuretools.task import get_container_settings, get_task_config


def main(
    job_id: str,
    pool_id: str,
    diseases: str | list[str],
    container_image_name: str = "pyrenew-hew",
    container_image_version: str = "latest",
    excluded_locations: list[str] = [
        "AS",
        "GU",
        "MO",
        "MP",
        "PR",
        "UM",
        "VI",
        "WY",
    ],
) -> None:
    """
    job_id
        Name for the Batch job.

    pool_id
        Azure Batch pool on which to run the job.

    diseases
        Name(s) of disease(s) to run as part of the job,
        as a single string (one disease) or a list of strings.
        Supported values are 'COVID-19' and 'Influenza'.

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

    excluded_locations
        List of two letter USPS location abbreviations to
        exclude from the job. Defaults to locations for which
        we typically do not have available NSSP ED visit data:
        ``["AS", "GU", "MO", "MP", "PR", "UM", "VI", "WY"]``.

    Returns
    -------
    None
    """
    supported_diseases = ["COVID-19", "Influenza"]

    disease_list = diseases

    invalid_diseases = set(disease_list) - set(supported_diseases)
    if invalid_diseases:
        raise ValueError(
            f"Unsupported diseases: {', '.join(invalid_diseases)}; "
            f"supported diseases are: {', '.join(supported_diseases)}"
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
        "--report-date {report_date} "
        "--exclude-last-n-days 5 "
        "--score "
        "--eval-data-path "
        "nssp_demo/nssp-archival-vintages/latest_comprehensive.parquet"
        "'"
    )

    locations = pl.read_csv(
        "https://www2.census.gov/geo/docs/reference/state.txt", separator="|"
    )

    all_locations = (
        locations.filter(~pl.col("STUSAB").is_in(excluded_locations))
        .get_column("STUSAB")
        .to_list()
    )

    for disease, state in itertools.product(disease_list, all_locations):
        task = get_task_config(
            f"{job_id}-{state}-{disease}-prod",
            base_call=base_call.format(
                state=state,
                disease=disease,
                report_date="latest",
            ),
            container_settings=container_settings,
        )
        client.task.add(job_id, task)

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

parser.add_argument(
    "--excluded-locations",
    type=str,
    help=(
        "Two-letter USPS location abbreviations to "
        "exclude from the job, as a whitespace-separated "
        "string. Defaults to a set of locations for which "
        "we typically do not have available NSSP ED visit "
        "data: 'AS GU MO MP PR UM VI WY'."
    ),
    default="AS GU MO MP PR UM VI WY",
)


if __name__ == "__main__":
    args = parser.parse_args()
    args.diseases = args.diseases.split()
    args.excluded_locations = args.excluded_locations.split()
    main(**vars(args))
