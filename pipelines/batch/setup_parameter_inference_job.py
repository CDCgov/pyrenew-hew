"""
Set up a multi-location,  multi-disease parameter
inference run for pyrenew-hew on Azure Batch.
"""

import argparse
import itertools
from pathlib import Path

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
    output_subdir: str | Path = "./",
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

     output_subdir
        Subdirectory of the output blob storage container
        in which to save results.

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

    pyrenew_hew_output_container = "pyrenew-test-output"
    n_warmup = 1000
    n_samples = 500

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
                "target": "/pyrenew-hew/nssp-etl",
            },
            {
                "source": "nssp-archival-vintages",
                "target": "/pyrenew-hew/nssp-archival-vintages",
            },
            {
                "source": "prod-param-estimates",
                "target": "/pyrenew-hew/params",
            },
            {
                "source": pyrenew_hew_output_container,
                "target": "/pyrenew-hew/output",
            },
            {
                "source": "pyrenew-hew-config",
                "target": "/pyrenew-hew/config",
            },
        ],
    )

    base_call = (
        "/bin/bash -c '"
        "python pipelines/forecast_state.py "
        "--disease {disease} "
        "--state {state} "
        "--n-training-days 450 "
        "--n-warmup {n_warmup} "
        "--n-samples {n_samples} "
        "--facility-level-nssp-data-dir nssp-etl/gold "
        "--state-level-nssp-data-dir "
        "nssp-archival-vintages/gold "
        "--param-data-dir params "
        "--output-dir {output_dir} "
        "--priors-path config/parameter_inference_priors.py "
        "--report-date {report_date} "
        "--exclude-last-n-days 5 "
        "--no-score "
        "--eval-data-path "
        "nssp-archival-vintages/latest_comprehensive.parquet"
        "'"
    )

    # to be replaced by forecasttools-py table
    locations = pl.read_csv(
        "https://www2.census.gov/geo/docs/reference/state.txt", separator="|"
    )

    all_locations = [
        loc
        for loc in ["US"] + locations.get_column("STUSAB").to_list()
        if loc not in excluded_locations
    ]

    for disease, state in itertools.product(disease_list, all_locations):
        task = get_task_config(
            f"{job_id}-{state}-{disease}-prod",
            base_call=base_call.format(
                state=state,
                disease=disease,
                report_date="latest",
                n_warmup=n_warmup,
                n_samples=n_samples,
                output_dir=str(Path("output", output_subdir)),
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
    "--output-subdir",
    type=str,
    help=(
        "Subdirectory of the output blob storage container "
        "in which to save results."
    ),
    default="./",
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
