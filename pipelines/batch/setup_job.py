"""
Set up a multi-location, multi-disease run
of pyrenew-hew on Azure Batch.
"""

import argparse
import itertools
from pathlib import Path

from azure.batch import models
from azure.identity import DefaultAzureCredential
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job
from azuretools.task import get_container_settings, get_task_config
from forecasttools import location_table

from pyrenew_hew.utils import validate_hew_letters

# Locations that are always excluded due to lack of NSSP ED visit data
DEFAULT_EXCLUDED_LOCATIONS = ["AS", "GU", "MP", "PR", "UM", "VI"]


def main(
    model_letters: str,
    job_id: str,
    pool_id: str,
    model_family: str,
    diseases: str | list[str],
    output_subdir: str | Path = "./",
    additional_forecast_letters: str = "",
    container_image_name: str = "pyrenew-hew",
    container_image_version: str = "latest",
    n_training_days: int = 150,
    exclude_last_n_days: int = 1,
    locations_include: list[str] | None = None,
    locations_exclude: list[str] | None = None,
    test: bool = False,
    vm_identity: bool = False,
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

    container_image_name
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
        Default 150.

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
        List of additional two letter USPS location abbreviations to
        exclude from the job. These will be combined with the default
        exclusions (see DEFAULT_EXCLUDED_LOCATIONS). If ``None``,
        only the default locations will be excluded. Default ``None``.

    test
        Is this a testing run? Default ``False``.

    model_family
        The model family to use for the job. Default 'pyrenew'.
        Supported values are 'pyrenew' and 'timeseries'.

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

    validate_hew_letters(model_letters)
    validate_hew_letters(additional_forecast_letters)

    if model_family == "timeseries" and model_letters != "e":
        raise ValueError(
            "Only model_letters 'e' is supported for the 'timeseries' model_family."
        )

    pyrenew_hew_output_container = (
        "pyrenew-test-output" if test else "pyrenew-hew-prod-output"
    )
    n_warmup = 200 if test else 1000
    n_samples = 200 if test else 500

    #TODO: Use VM managed identity with DefaultAzureCredential()
    creds = EnvCredentialHandler()
    client = get_batch_service_client(creds)
    job = models.JobAddParameter(
        id=job_id,
        pool_info=models.PoolInformation(pool_id=pool_id),
    )
    create_job(client, job)

    container_image = (
        f"ghcr.io/cdcgov/{container_image_name}:{container_image_version}"
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
            {
                "source": "nwss-vintages",
                "target": "/pyrenew-hew/nwss-vintages",
            },
        ],
    )

    if model_family == "pyrenew":
        run_script = "forecast_pyrenew.py"
        additional_args = (
            f"--n-warmup {n_warmup} "
            "--nwss-data-dir nwss-vintages "
            "--priors-path pipelines/priors/prod_priors.py "
            f"--additional-forecast-letters {additional_forecast_letters} "
        )
    elif model_family == "timeseries":
        run_script = "forecast_timeseries.py"
        additional_args = ""
    else:
        raise ValueError(
            f"Unsupported model family: {model_family}. "
            "Supported values are 'pyrenew' and 'timeseries'."
        )

    base_call = (
        "/bin/bash -c '"
        f"uv run python pipelines/{run_script} "
        "--disease {disease} "
        "--loc {loc} "
        f"--n-training-days {n_training_days} "
        f"--n-samples {n_samples} "
        "--facility-level-nssp-data-dir nssp-etl/gold "
        "--state-level-nssp-data-dir "
        "nssp-archival-vintages/gold "
        "--param-data-dir params "
        "--output-dir {output_dir} "
        "--credentials-path config/creds.toml "
        "--report-date {report_date} "
        f"--exclude-last-n-days {exclude_last_n_days} "
        f"--model-letters {model_letters} "
        "--eval-data-path "
        "nssp-etl/latest_comprehensive.parquet "
        f"{additional_args}"
        "'"
    )

    loc_abbrs = location_table.get_column("short_name").to_list()
    locations_include = locations_include or loc_abbrs

    # Always exclude the default locations
    locations_exclude = locations_exclude or []
    # Combine default exclusions with any additional exclusions
    all_exclusions = list(set(DEFAULT_EXCLUDED_LOCATIONS + locations_exclude))

    all_locations = [
        loc
        for loc in loc_abbrs
        if loc not in all_exclusions and loc in locations_include
    ]

    for disease, loc in itertools.product(disease_list, all_locations):
        task = get_task_config(
            f"{job_id}-{loc}-{disease}-prod",
            base_call=base_call.format(
                loc=loc,
                disease=disease,
                report_date="latest",
                output_dir=str(Path("output", output_subdir)),
            ),
            container_settings=container_settings,
            log_blob_container="pyrenew-hew-logs",
            log_blob_account=creds.azure_blob_storage_account,
            log_subdir=job_id,
            log_compute_node_identity_reference=(
                creds.compute_node_identity_reference
            ),
        )
        client.task.add(job_id, task)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--model_letters",
        type=str,
        help=(
            "Fit the model corresponding to the provided model letters (e.g. 'he', 'e', 'hew')."
        ),
    )

    parser.add_argument(
        "--job_id", type=str, help="Name for the Azure batch job"
    )
    parser.add_argument(
        "--pool_id",
        type=str,
        help=("Name of the Azure batch pool on which to run the job"),
        default="pyrenew-pool",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        default="COVID-19 Influenza",
        help=(
            "Name(s) of disease(s) to run as part of the job, "
            "as a whitespace-separated string. Supported "
            "values are 'COVID-19' and 'Influenza'. "
            "Default 'COVID-19 Influenza' (i.e. run for both)."
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
        default=150,
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
            "string. If not set, include all "
            "available locations except any explicitly excluded "
            "via --locations-exclude."
        ),
        default=None,
    )

    parser.add_argument(
        "--locations-exclude",
        type=str,
        help=(
            "Additional two-letter USPS location abbreviations to "
            "exclude from the job, as a whitespace-separated "
            "string. These will be combined with the default "
            f"exclusions ({' '.join(DEFAULT_EXCLUDED_LOCATIONS)}) which are always excluded."
        ),
        default=None,
    )

    parser.add_argument(
        "--additional-forecast-letters",
        type=str,
        help=(
            "Forecast the following signals even if they were not fit. "
            "Fit signals are always forecast."
        ),
        default=None,
    )

    parser.add_argument(
        "--model-family",
        type=str,
        help=(
            "Model family to use for the job. "
            "Supported values are 'pyrenew' and 'timeseries'. "
            "Default 'pyrenew'."
        ),
        default="pyrenew",
    )

    args = parser.parse_args()
    args.diseases = args.diseases.split()
    args.locations_include = (
        args.locations_include.split() if args.locations_include else []
    )
    args.locations_exclude = (
        args.locations_exclude.split() if args.locations_exclude else []
    )
    if args.additional_forecast_letters is None:
        args.additional_forecast_letters = args.model_letters
    main(**vars(args))
