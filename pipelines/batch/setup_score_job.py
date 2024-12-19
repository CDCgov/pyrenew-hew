"""
Set up a multi-location, multi-disease production run
of pyrenew-hew on Azure Batch.
"""

import argparse
import logging
import os
from pathlib import Path

import polars as pl
from azure.batch import models
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job_if_not_exists
from azuretools.task import get_container_settings, get_task_config

from pipelines.utils import get_all_forecast_dirs


def main(
    job_id: str,
    pool_id: str,
    dirs_to_score: list[Path] | list[str],
    container_image_name: str = "pyrenew-hew",
    container_image_version: str = "latest",
) -> None:
    """
    job_id
        Name for the Batch job.

    pool_id
        Azure Batch pool on which to run the job.

    dirs_to_score
        Directories to containing forecasts to be scored.

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
        Sets up the job and task as side effects.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Creating scoring jobs for the following directories: "
        f"{dirs_to_score}"
    )

    pyrenew_hew_output_container = "pyrenew-hew-prod-output"

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
                "source": "nssp-archival-vintages",
                "target": "/pyrenew-hew/nssp-archival-vintages",
            },
            {
                "source": pyrenew_hew_output_container,
                "target": "/pyrenew-hew/output",
            },
        ],
    )

    base_call = (
        "/bin/bash -c '"
        "python pipelines/score_location.py "
        "{model_batch_dir_path} "
        "nssp-archival-vintages/latest_comprehensive.parquet "
        "--state {location}"
        "'"
    )

    locations = pl.read_csv(
        "https://www2.census.gov/geo/docs/reference/state.txt", separator="|"
    )

    loc_abbrs = locations.get_column("STUSAB").to_list() + ["US"]

    for score_dir in dirs_to_score:
        forecast_dirs = get_all_forecast_dirs(
            score_dir, ["COVID-19", "Influenza"]
        )
        for model_batch_dir in forecast_dirs:
            location_names = [
                f.name
                for f in os.scandir(
                    Path(score_dir, model_batch_dir, "model_runs")
                )
                if f.is_dir() and f.name in loc_abbrs
            ]
            for location in location_names:
                model_batch_dir_path = f"output/{score_dir}/{model_batch_dir}"
                task = get_task_config(
                    f"{model_batch_dir}-{location}",
                    base_call=base_call.format(
                        model_batch_dir_path=model_batch_dir_path,
                        location=location,
                    ),
                    container_settings=container_settings,
                )
                client.task.add(job_id, task)
            pass  # end loop over locations
        pass  # end loop over forecast dirs
    pass  # end loop over dirs_to_score

    return None


parser = argparse.ArgumentParser()

parser.add_argument("job_id", type=str, help="Name for the Azure batch job")
parser.add_argument(
    "pool_id",
    type=str,
    help=("Name of the Azure batch pool on which to run the job"),
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
    "dirs_to_score",
    nargs="*",
    type=Path,
    help=("local paths to directories to score"),
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
