#!/usr/bin/env -S uv run --script
# PEP 723 dependency definition: https://peps.python.org/pep-0723/
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#    "dagster-azure>=0.27.4",
#    "dagster-docker>=0.27.4",
#    "dagster-postgres>=0.27.4",
#    "dagster-webserver==1.12.2",
#    "dagster==1.12.2",
#    "cfa-dagster @ git+https://github.com/cdcgov/cfa-dagster.git",
#    "pyyaml>=6.0.2",
# ]
# ///
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import dagster as dg
from cfa_dagster.azure_batch.executor import azure_batch_executor
from cfa_dagster.azure_container_app_job.executor import (
    azure_container_app_job_executor as azure_caj_executor,
)
from cfa_dagster.docker.executor import docker_executor
from cfa_dagster.utils import collect_definitions
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

class PyrenewAssetConfig(dg.Config):
    # when using the docker_executor, specify the image you'd like to use
    image: str = f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}"

#
def build_pyrenew_asset(
    model_letters: str,
    model_family: str = "pyrenew",
    asset_name: str = str(None),
    depends_on: list[str] = None,
):
    # Partition Definitions
    full_disease_list = ["COVID-19", "Influenza", "RSV"]

    full_state_list = [
        'US', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO',
        'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID',
        'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
        'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
        'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
        'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
        'WV', 'WI', 'WY', 'AS', 'GU', 'MP', 'PR',
        'UM', 'VI'
    ]
    # TODO: encode a way for people to customize excluded locations
    DEFAULT_EXCLUDED_LOCATIONS: list[str] = ["AS", "GU", "MP", "PR", "UM", "VI"]
    disease_list: list[str] = full_disease_list
    state_list: list[str] = [state for state in full_state_list if state not in DEFAULT_EXCLUDED_LOCATIONS]

    # w models do not forecast RSV or Influenza
    # e models do not forecast WY as a location
    if "w" in model_letters:
        disease_list.remove("RSV")
        disease_list.remove("Influenza")
    elif "e" in model_letters:
        state_list.remove("WY")

    disease_partitions = dg.StaticPartitionsDefinition(disease_list)
    state_partitions = dg.StaticPartitionsDefinition(state_list)
    two_dimensional_pyrenew_partition = dg.MultiPartitionsDefinition(
        {"disease": disease_partitions, "loc": state_partitions}
    )

    if depends_on is None:
        depends_on = []

    @dg.asset(
        partitions_def=two_dimensional_pyrenew_partition,
        name=asset_name,
        deps=depends_on
    )
    def pyrenew_asset(
        context: dg.AssetExecutionContext,
    ) -> str:
        keys_by_dimension: dg.MultiPartitionKey = context.partition_key.keys_by_dimension
        disease = keys_by_dimension["disease"]
        loc = keys_by_dimension["loc"]
        n_training_days = 150
        n_samples = 500
        exclude_last_n_days = 1
        n_warmup = 1000
        additional_forecast_letters = model_letters
        forecast_date = str(date.today())
        output_subdir = f"{forecast_date}_forecasts"
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
            f"VIRTUAL_ENV=.venv && "
            f"uv run python pipelines/{run_script} "
            f"--disease {disease} "
            f"--loc {loc} "
            f"--n-training-days {n_training_days} "
            f"--n-samples {n_samples} "
            "--facility-level-nssp-data-dir nssp-etl/gold "
            "--state-level-nssp-data-dir nssp-archival-vintages/gold "
            "--param-data-dir params "
            f"--output-dir test-output/{output_subdir} "
            "--credentials-path config/creds.toml "
            f"--report-date latest "
            f"--exclude-last-n-days {exclude_last_n_days} "
            f"--model-letters {model_letters} "
            "--eval-data-path "
            "nssp-etl/latest_comprehensive.parquet "
            f"{additional_args}"
            "'"
        )
        run = subprocess.run(base_call, shell=True, check=True)
        return asset_name
    return pyrenew_asset

# Use the builder to create multiple assets
timeseries_e_output = build_pyrenew_asset(
    model_letters="e", asset_name="timeseries_e_output", model_family="timeseries"
)
pyrenew_h_output = build_pyrenew_asset(
    model_letters="h", asset_name="pyrenew_h_output"
)
pyrenew_e_output = build_pyrenew_asset(
    model_letters="e", asset_name="pyrenew_e_output", depends_on=["timeseries_e_output"]
)
pyrenew_he_output = build_pyrenew_asset(
    model_letters="he", asset_name="pyrenew_he_output", depends_on=["timeseries_e_output"]
)
pyrenew_hw_output = build_pyrenew_asset(
    model_letters="hw", asset_name="pyrenew_hw_output"
)
pyrenew_hew_output = build_pyrenew_asset(
    model_letters="hew", asset_name="pyrenew_hew_output", depends_on=["timeseries_e_output"]
)

# Sample assets - useful for reference and testing
@dg.asset(
    kinds={"azure_blob"},
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
    subprocess.run("Rscript hello.R", shell=True, check=True)

    # Read the random number from output.txt
    with open("output.txt", "r") as f:
        random_number = f.read().strip()

    return dg.MaterializeResult(
        metadata={
            # add metadata from upstream asset
            "config": dg.MetadataValue.json(json.loads(basic_blob_asset)),
            # Dagster will plot numeric values as you repeat runs
            "output_value": dg.MetadataValue.int(int(random_number)),
        }
    )

disease_partitions = dg.StaticPartitionsDefinition(["COVID", "FLU", "RSV"])

@dg.asset(
    description="A partitioned asset that runs R code for different diseases",
    partitions_def=disease_partitions,
)
def partitioned_r_asset(context: dg.OpExecutionContext):
    disease = context.partition_key
    subprocess.run(f"Rscript hello.R {disease}", shell=True, check=True)

workdir = "pyrenew-hew"
local_workdir = Path(__file__).parent.resolve()

# add this to a job or the Definitions class to use it
docker_executor_configured = docker_executor.configured(
    {
        # specify a default image
        "image": f"pyrenew-hew:dagster_latest_{user}",
        "env_vars": [f"DAGSTER_USER={user}","VIRTUAL_ENV=/pyrenew-hew/.dg_venv"],
        "container_kwargs": {
            "volumes": [
                # bind the ~/.azure folder for optional cli login
                f"/home/{user}/.azure:/root/.azure",
                # bind current file so we don't have to rebuild
                # the container image for workflow changes
                f"{__file__}:/{workdir}/{os.path.basename(__file__)}",
                # blob container mounts for pyrenew-hew
                f"/{local_workdir}/nssp-etl:/pyrenew-hew/nssp-etl",
                f"/{local_workdir}/nssp-archival-vintages:/pyrenew-hew/nssp-archival-vintages",
                f"/{local_workdir}/nwss-vintages:/pyrenew-hew/nwss-vintages",
                f"/{local_workdir}/params:/pyrenew-hew/params",
                f"/{local_workdir}/config:/pyrenew-hew/config",
                f"/{local_workdir}/output:/pyrenew-hew/output",
                f"/{local_workdir}/test-output:/pyrenew-hew/test-output",
            ]
        },
    }
)

# configuring an executor to run workflow steps on Azure Container App Jobs
# add this to a job or the Definitions class to use it
azure_caj_executor_configured = azure_caj_executor.configured(
    {
        "image": f"cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest_{user}",
        "env_vars": [f"DAGSTER_USER={user}","VIRTUAL_ENV=/pyrenew-hew/.dg_venv"],
    }
)

# configuring an executor to run workflow steps on Azure Batch 4CPU 16GB RAM pool
# add this to a job or the Definitions class to use it
azure_batch_executor_configured = azure_batch_executor.configured(
    {   "pool_name": "pyrenew-pool",
        "image": f"cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest_{user}",
        "env_vars": [f"DAGSTER_USER={user}","VIRTUAL_ENV=/pyrenew-hew/.dg_venv"],
        "container_kwargs": {
            "volumes": [
                # bind the ~/.azure folder for optional cli login
                # f"/home/{user}/.azure:/root/.azure",
                # bind current file so we don't have to rebuild
                # the container image for workflow changes
                # blob container mounts for pyrenew-hew
                "nssp-archival-vintages:/pyrenew-hew/nssp-archival-vintages",
                "nssp-etl:/pyrenew-hew/nssp-etl",
                "nwss-vintages:/pyrenew-hew/nwss-vintages",
                "prod-param-estimates:/pyrenew-hew/params",
                "pyrenew-hew-config:/pyrenew-hew/config",
                "pyrenew-hew-prod-output:/pyrenew-hew/output",
                "pyrenew-test-output:/pyrenew-hew/test-output",
            ],
            "working_dir":"/pyrenew-hew",
        },
    }
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

# schedule the job to run weekly
# schedule_every_wednesday = dg.ScheduleDefinition(
#     name="weekly_cron", cron_schedule="0 9 * * 3",
# )

pyrenew_asset_job = dg.define_asset_job(
    name="pyrenew_asset_job",
    executor_def=azure_batch_executor_configured,
    selection=dg.AssetSelection.assets(
        "timeseries_e_output"
    ),
    # tag the run with your user to allow for easy filtering in the Dagster UI
    tags={"user": user},
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

schedule_every_wednesday = dg.ScheduleDefinition(
    name="weekly_cron",
    cron_schedule="0 9 * * 3",
    job=pyrenew_asset_job
)

# env variable set by Dagster CLI
is_production = os.getenv("DAGSTER_IS_DEV_CLI", "false") == "false"
# change storage accounts between dev and prod
storage_account = "cfadagster" if is_production else "cfadagsterdev"

# collect Dagster definitions from the current file
collected_defs = collect_definitions(globals())

# Create Definitions object
defs = dg.Definitions(
    assets=collected_defs["assets"],
    asset_checks=collected_defs["asset_checks"],
    jobs=collected_defs["jobs"],
    sensors=collected_defs["sensors"],
    schedules=collected_defs["schedules"],
    resources=resources_def,
    # setting Docker as the default executor. comment this out to use
    # the default executor that runs directly on your computer
    # executor=docker_executor_configured,
    # executor=azure_caj_executor_configured,
    executor=azure_batch_executor_configured,
)
