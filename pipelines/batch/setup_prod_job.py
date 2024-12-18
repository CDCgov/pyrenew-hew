"""
Set up a multi-location, multi-disease production run
of pyrenew-hew on Azure Batch.
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
    n_training_days: int = 90,
    exclude_last_n_days: int = 1,
    locations_include: list[str] = None,
    locations_exclude: list[str] = [
        "AS",
        "GU",
        "MO",
        "MP",
        "PR",
        "UM",
        "VI",
        "WY",
    ],
    test: bool = False,
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

    n_training_days
        Number of training days of data to use for model fitting.
        Default 90.

    exclude_last_n_days
        Number of days of available data to exclude from fitting.
        Default 1. Note that we start the lookback for the
        ``n_training_days`` of data after these exclusions,
        so there will always be ``n_training_days`` of observations
        for fitting; ``exclude_last_n_days`` determines where
        the date range of observations starts and ends.

    locations_include
        List of two-letter USPS location abbreviations for locations
        to include in the job (unless explicitly excluded by
        --locations-exclude). If ``None``, use all available
        not-explicitly-excluded locations. Default ``None``.

    locations_exclude
        List of two letter USPS location abbreviations to
        exclude from the job. If ``None``, do not exclude any
        locations. Defaults to a list of locations for which
        we typically do not have available NSSP ED visit data:
        ``["AS", "GU", "MO", "MP", "PR", "UM", "VI", "WY"]``.

    test
        Is this a testing run? Default ``False``.

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

    pyrenew_hew_output_container = (
        "pyrenew-test-output" if test else "pyrenew-hew-prod-output"
    )
    n_warmup = 200 if test else 1000
    n_samples = 200 if test else 500

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
        "python pipelines/workflows/forecast_state.py "
        "--disease {disease} "
        "--state {state} "
        "--n-training-days {n_training_days} "
        "--n-warmup {n_warmup} "
        "--n-samples {n_samples} "
        "--facility-level-nssp-data-dir nssp-etl/gold "
        "--state-level-nssp-data-dir "
        "nssp-archival-vintages/gold "
        "--param-data-dir params "
        "--output-dir {output_dir} "
        "--priors-path pipelines/priors/prod_priors.py "
        "--report-date {report_date} "
        "--exclude-last-n-days {exclude_last_n_days} "
        "--no-score "
        "--eval-data-path "
        "nssp-archival-vintages/latest_comprehensive.parquet"
        "'"
    )

    # to be replaced by forecasttools-py table
    locations = pl.read_csv(
        "https://www2.census.gov/geo/docs/reference/state.txt", separator="|"
    )

    loc_abbrs = locations.get_column("STUSAB").to_list() + ["US"]
    if locations_include is None:
        locations_include = loc_abbrs
    if locations_exclude is None:
        locations_exclude = []

    all_locations = [
        loc
        for loc in loc_abbrs
        if loc not in locations_exclude and loc in locations_include
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
                n_training_days=n_training_days,
                exclude_last_n_days=exclude_last_n_days,
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
    "--n-training-days",
    type=int,
    help=(
        "Number of 'training days' of observed data "
        "to use for model fitting."
    ),
    default=90,
)

parser.add_argument(
    "--exclude-last-n-days",
    type=int,
    help=(
        "Number of days to drop from the end of the timeseries "
        "of observed data when constructing the training data."
    ),
    default=1,
)

parser.add_argument(
    "--locations-include",
    type=str,
    help=(
        "Two-letter USPS location abbreviations to "
        "include in the job, as a whitespace-separated "
        "string. If not set, include all ",
        "available locations except any explicitly excluded "
        "via --locations-exclude.",
    ),
    default=None,
)

parser.add_argument(
    "--locations-exclude",
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
    if args.locations_include is not None:
        args.locations_include = args.locations_include.split()
    args.locations_exclude = args.locations_exclude.split()
    main(**vars(args))
