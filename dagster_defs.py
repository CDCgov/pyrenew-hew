#!/usr/bin/env -S uv run --script
# PEP 723 dependency definition: https://peps.python.org/pep-0723/
# /// script
# requires-python = ">=3.13"
# dependencies = [
#    "dagster-azure>=0.27.4",
#    "dagster-docker>=0.27.4",
#    "dagster-postgres>=0.27.4",
#    "dagster-webserver",
#    "dagster==1.11.4",
#    "cfa-dagster @ git+https://github.com/cdcgov/cfa-dagster.git",
#    "pyyaml>=6.0.2",
# ]
# ///
import os
import subprocess
import sys
from pathlib import Path
import json

import dagster as dg
from cfa_dagster.azure_batch.executor import azure_batch_executor
from cfa_dagster.azure_container_app_job.executor import (
    azure_container_app_job_executor as azure_caj_executor,
)
from cfa_dagster.docker.executor import docker_executor
from dagster_azure.adls2 import (
    ADLS2DefaultAzureCredential,
    ADLS2PickleIOManager,
    ADLS2Resource,
)
from dagster_azure.blob import (
    AzureBlobStorageDefaultCredential,
    AzureBlobStorageResource,
)

# Start the Dagster UI and set necessary env vars
if "--dev" in sys.argv:
    # Set environment variables
    home_dir = Path.home()
    dagster_user = home_dir.name
    dagster_home = home_dir / ".dagster_home"

    os.environ["DAGSTER_USER"] = dagster_user
    os.environ["DAGSTER_HOME"] = str(dagster_home)
    script = sys.argv[0]

    # Run the Dagster webserver
    try:
        subprocess.run(["dagster", "dev", "-f", script])
    except KeyboardInterrupt:
        print("\nShutting down cleanly...")


# get the user from the environment, throw an error if variable is not set
user = os.environ["DAGSTER_USER"]


@dg.asset(
    kinds={'azure_blob'},
    description="An asset that downloads a file from Azure Blob Storage",
)
def basic_blob_asset(azure_blob_storage: AzureBlobStorageResource):
    """
    An asset that downloads a config file from Azure Blob
    """
    container_name = "cfadagsterdev"
    with azure_blob_storage.get_client() as blob_storage_client:
        container_client = blob_storage_client.get_container_client(container_name)
    downloader = container_client.download_blob("test-files/test_config.json")
    print("Downloaded file from blob!")
    return downloader.readall().decode("utf-8")


@dg.asset(
    description="An asset that runs R code",
)
def basic_r_asset(basic_blob_asset):
    run = subprocess.run("Rscript hello.R", shell=True)
    # throw an exception if the script returns an error code
    run.check_returncode()

    # Read the random number from output.txt
    with open("output.txt", "r") as f:
        random_number = f.read().strip()

    return dg.MaterializeResult(
        metadata={
            # add metadata from upstream asset
            "config": dg.MetadataValue.json(json.loads(basic_blob_asset)),
            # Dagster will plot numeric values as you repeat runs
            "output_value": dg.MetadataValue.int(int(random_number))
        }
    )


disease_partitions = dg.StaticPartitionsDefinition(['COVID', 'FLU', 'RSV'])


@dg.asset(
    description="A partitioned asset that runs R code for different diseases",
    partitions_def=disease_partitions,
)
def partitioned_r_asset(context: dg.OpExecutionContext):
    disease = context.partition_key
    run = subprocess.run(f"Rscript hello.R {disease}", shell=True)
    # throw an exception if the script returns an error code
    run.check_returncode()


# configuring an executor to run workflow steps on Docker
# add this to a job or the Definitions class to use it
docker_executor_configured = docker_executor.configured(
    {
        # specify a default image
        "image": "basic-r-asset",
        "env_vars": [f"DAGSTER_USER={user}"],
        "container_kwargs": {
            "volumes": [
                # bind the ~/.azure folder for optional cli login
                f"/home/{user}/.azure:/root/.azure",
                # bind current file so we don't have to rebuild
                # the container image for workflow changes
                f"{__file__}:/app/{os.path.basename(__file__)}",
            ]
        },
    }
)

# configuring an executor to run workflow steps on Azure Container App Jobs
# add this to a job or the Definitions class to use it
azure_caj_executor_configured = azure_caj_executor.configured(
    {
        "image": f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}",
        "env_vars": [f"DAGSTER_USER={user}"],
    }
)

# configuring an executor to run workflow steps on Azure Batch 4CPU 16GB RAM pool
# add this to a job or the Definitions class to use it
azure_batch_executor_configured = azure_batch_executor.configured(
    {
        "image": f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}",
        "env_vars": [f"DAGSTER_USER={user}"],
        # "container_kwargs": {
        #     # default working_dir is /app for Batch
        #     # I have not been able to get Batch to work with other dirs
        #     "working_dir": "/app",
        # },
    }
)

# jobs are used to materialize assets with a given configuration
basic_r_asset_job = dg.define_asset_job(
    name="basic_r_asset_job",
    # specify an executor including docker, Azure Container App Job, or
    # the future Azure Batch executor
    executor_def=docker_executor_configured,
    # uncomment the below to switch to run on Azure Container App Jobs.
    # remember to rebuild and push your image if you made any workflow changes
    # executor_def=azure_caj_executor_configured,
    selection=dg.AssetSelection.assets(basic_r_asset),
    # tag the run with your user to allow for easy filtering in the Dagster UI
    tags={"user": user},
)

partitioned_r_asset_job = dg.define_asset_job(
    name="partitioned_r_asset_job",
    executor_def=docker_executor_configured,
    # uncomment the below to switch to run on Azure Container App Jobs.
    # remember to rebuild and push your image if you made any workflow changes
    # executor_def=azure_caj_executor_configured,
    selection=dg.AssetSelection.assets(partitioned_r_asset),
    # tag the run with your user to allow for easy filtering in the Dagster UI
    tags={"user": user},
)

# schedule the job to run weekly
schedule_every_wednesday = dg.ScheduleDefinition(
    name="weekly_cron", cron_schedule="0 9 * * 3", job=basic_r_asset_job
)


# this prefix allows your assets to be stored in Azure
# without conflicting with other users
adls2_prefix = f"dagster-files/{user}/"

resources_def = {
    # This IOManager lets Dagster serialize asset outputs and store them
    # in Azure to pass between assets
    "io_manager": ADLS2PickleIOManager(
        adls2_file_system="cfadagsterdev",
        adls2_prefix=adls2_prefix,
        adls2=ADLS2Resource(
            storage_account="cfadagsterdev",
            credential=ADLS2DefaultAzureCredential(kwargs={}),
        ),
        lease_duration=-1,  # unlimited lease for writing large files
    ),
    "azure_blob_storage": AzureBlobStorageResource(
        account_url="cfadagsterdev.blob.core.windows.net",
        credential=AzureBlobStorageDefaultCredential(),
    ),
}


# Add assets, jobs, schedules, and sensors here to have them appear in the
# Dagster UI
defs = dg.Definitions(
    assets=[basic_r_asset, partitioned_r_asset, basic_blob_asset],
    jobs=[basic_r_asset_job, partitioned_r_asset_job],
    schedules=[schedule_every_wednesday],
    resources=resources_def,
    # setting Docker as the default executor. comment this out to use
    # the default executor that runs directly on your computer
    executor=docker_executor_configured,
    # executor=azure_caj_executor_configured,
    # executor=azure_batch_executor_configured,
)
