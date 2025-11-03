#!/usr/bin/env -S uv run --script
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
from datetime import date
from pathlib import Path

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

# end_offset=1 allows you to target the current month
# by default, dagster only allows you to materialize after a month is complete
monthly_partition = dg.MonthlyPartitionsDefinition(
    start_date="2024-01-01", end_offset=1
)

# get the user from the environment, throw an error if variable is not set
user = os.environ["DAGSTER_USER"]
class PyrenewAssetConfig(dg.Config):
    # when using the docker_executor, specify the image you'd like to use
    image: str = f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}"

# TODO: Encode business rules for exclusions across the different model letters and types
disease_list = ["COVID-19", "Influenza", "RSV"]
disease_partitions = dg.StaticPartitionsDefinition(disease_list)
state_list = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT",
    "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY",
    "DC", "US"
]
state_partitions = dg.StaticPartitionsDefinition(state_list)
two_dimensional_partitions = dg.MultiPartitionsDefinition(
    {"disease": disease_partitions, "loc": state_partitions}
)
class PyrenewHOutputConfig(dg.Config):
    # when using the docker_executor, specify the image you'd like to use
    image: str = "pyrenew-hew:dagster_latest"


def build_pyrenew_asset(
    model_letters: str,
    model_family: str = "pyrenew",
    partitions_def: dg.PartitionsDefinition = two_dimensional_partitions,
    asset_name: str = str(None),
    depends_on: list[str] = None,
):
    if depends_on is None:
        depends_on = []
    @dg.asset(
        partitions_def=partitions_def,
        name=asset_name,
        deps=depends_on
    )
    def pyrenew_asset(
        context: dg.AssetExecutionContext,
        config: PyrenewHOutputConfig,
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

workdir = "pyrenew-hew"
local_workdir = Path(__file__).parent.resolve()

# add this to a job or the Definitions class to use it
docker_executor_configured = docker_executor.configured(
    {
        # specify a default image
        "image": "pyrenew-hew:dagster_latest",
        "env_vars": [f"DAGSTER_USER={user}"],
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
        "image": "cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest",
        "env_vars": [f"DAGSTER_USER={user}"],
    }
)

# configuring an executor to run workflow steps on Azure Batch 4CPU 16GB RAM pool
# add this to a job or the Definitions class to use it
azure_batch_executor_configured = azure_batch_executor.configured(
    {
        "image": f"cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest_{user}",
        "env_vars": [f"DAGSTER_USER={user}"],
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
                "params:/pyrenew-hew/params",
                "config:/pyrenew-hew/config",
                "output:/pyrenew-hew/output",
                "test-output:/pyrenew-hew/test-output",
            ],
            "working_dir":"/app",
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

# Add assets, jobs, schedules, and sensors here to have them appear in the
# Dagster UI
defs = dg.Definitions(
    assets=[
        # nssp_gold,
        # nssp_latest_comprehensive,
        # nwss_gold,
        # nhsn_latest,
        timeseries_e_output,
        pyrenew_e_output,
        pyrenew_he_output,
        pyrenew_hew_output,
        pyrenew_h_output,
        pyrenew_hw_output
    ],
    jobs=[],
    resources=resources_def,
    # setting Docker as the default executor. comment this out to use
    # the default executor that runs directly on your computer
    executor=docker_executor_configured,
    # executor=azure_caj_executor_configured,
    # executor=azure_batch_executor_configured,
)
