import os
import subprocess
from datetime import date
from pathlib import Path

import dagster as dg
from cfa_dagster import (
    ADLS2PickleIOManager,
    AzureContainerAppJobRunLauncher,
    azure_batch_executor,
    collect_definitions,
    docker_executor,
    start_dev_env,
)
from cfa_dagster import (
    azure_container_app_job_executor as azure_caj_executor,
)
from dagster_azure.blob import (
    AzureBlobStorageDefaultCredential,
    AzureBlobStorageResource,
)

# ---------------------- #
# Dagster Initialization
# ---------------------- #

# function to start the dev server
start_dev_env(__name__)

user = os.getenv("DAGSTER_USER")

# --------------------------------------------------------------- #
# Partitions: how are the data split and processed in Azure Batch?
# --------------------------------------------------------------- #
# fmt: off
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
# fmt: off

# Disease Partitions
disease_partitions = dg.StaticPartitionsDefinition(["COVID-19", "Influenza", "RSV"])

# State Partitions
state_partitions = dg.StaticPartitionsDefinition(full_state_list)

# Multi Partitions
multi_partition_def = dg.MultiPartitionsDefinition(
    {"disease": disease_partitions, "state": state_partitions}
)

# ----------------------------------------------------------- #
# Asset Definitions - What are we outputting in our pipeline?
# ----------------------------------------------------------- #

# ---------------
# Worker Function
# ---------------


# This function is NOT an asset itself, but is called by assets to run the pyrenew model
def run_pyrenew_model(
    context: dg.AssetExecutionContext,
    model_letters: str = "h",
    model_family: str = "pyrenew",
):
    # Parsing partitions into call parameters for the job
    keys_by_dimension: dg.MultiPartitionKey = context.partition_key.keys_by_dimension
    DEFAULT_EXCLUDED_LOCATIONS: list[str] = ["AS", "GU", "MP", "PR", "UM", "VI"]
    disease = keys_by_dimension["disease"]
    state = keys_by_dimension["state"]

    # Exclusions
    if state in DEFAULT_EXCLUDED_LOCATIONS:
        context.log.info(
            f"Location {state} is in the default excluded locations. Skipping model run."
            f"Excluded locations: {DEFAULT_EXCLUDED_LOCATIONS}"
            "Future logic may be customizable."
        )
        return
    if "w" in model_letters and disease != "COVID-19":
        context.log.info(
            f"Model letter 'w' is only applicable for COVID-19. Skipping model run for disease {disease}."
        )
        return
    if "e" in model_letters and state == "WY":
        context.log.info(
            "Model letter 'e' is not applicable for location WY. Skipping model run."
        )
        return

    # TODO: Make configurable in the UI
    n_training_days = 150
    n_samples = 500
    exclude_last_n_days = 1
    n_warmup = 1000
    additional_forecast_letters = model_letters
    forecast_date = str(date.today())
    # TODO: parameterize this for dagster
    output_dir = "test-output"
    output_subdir = f"{forecast_date}_forecasts"
    full_output_dir = f"{output_dir}/{output_subdir}"
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
        f"--loc {state} "
        f"--n-training-days {n_training_days} "
        f"--n-samples {n_samples} "
        "--facility-level-nssp-data-dir nssp-etl/gold "
        "--state-level-nssp-data-dir nssp-archival-vintages/gold "
        "--param-data-dir params "
        f"--output-dir {full_output_dir} "
        "--credentials-path config/creds.toml "
        f"--report-date latest "
        f"--exclude-last-n-days {exclude_last_n_days} "
        f"--model-letters {model_letters} "
        "--eval-data-path "
        "nssp-etl/latest_comprehensive.parquet "
        f"{additional_args}"
        "'"
    )
    subprocess.run(base_call, shell=True, check=True)


# --------------------------------------------------------------------
# Assets: these are the core of Dagster - functions that specify data
# --------------------------------------------------------------------

# Upstream Data #

@dg.asset
def nhsn_data(context: dg.AssetExecutionContext):
    return "nhsn_data"


@dg.asset
def nssp_gold(context: dg.AssetExecutionContext):
    return "nssp_gold"


@dg.asset
def nssp_latest_comprehensive(context: dg.AssetExecutionContext):
    return "nssp_latest_comprehensive"


@dg.asset
def nwss_data(context: dg.AssetExecutionContext):
    return "nwss_data"


# Pyrenew Assets #

# Timeseries E
nssp_deps = ["nssp_gold", "nssp_latest_comprehensive"]


@dg.asset(partitions_def=multi_partition_def, deps=nssp_deps)
def timeseries_e_output(context: dg.AssetExecutionContext):
    run_pyrenew_model(context, model_letters="e", model_family="timeseries")
    return "timeseries_e_output"


# Pyrenew E
@dg.asset(partitions_def=multi_partition_def, deps=["timeseries_e_output"] + nssp_deps)
def pyrenew_e_output(context: dg.AssetExecutionContext):
    run_pyrenew_model(context, model_letters="e", model_family="pyrenew")
    return "pyrenew_e_output"


# Pyrenew H
@dg.asset(partitions_def=multi_partition_def, deps=["nhsn_data"])
def pyrenew_h_output(context: dg.AssetExecutionContext):
    run_pyrenew_model(context, model_letters="h", model_family="pyrenew")
    return "pyrenew_h_output"


# Pyrenew HE
@dg.asset(
    partitions_def=multi_partition_def,
    deps=["timeseries_e_output", "nhsn_data"] + nssp_deps,
)
def pyrenew_he_output(context: dg.AssetExecutionContext):
    run_pyrenew_model(context, model_letters="he", model_family="pyrenew")
    return "pyrenew_he_output"


# Pyrenew HW
@dg.asset(partitions_def=multi_partition_def, deps=["nhsn_data", "nwss_data"])
def pyrenew_hw_output(context: dg.AssetExecutionContext):
    run_pyrenew_model(context, model_letters="hw", model_family="pyrenew")
    return "pyrenew_hw_output"


# Pyrenew HEW
@dg.asset(
    partitions_def=multi_partition_def,
    deps=["timeseries_e_output"] + nssp_deps + ["nhsn_data", "nwss_data"],
)
def pyrenew_hew_output(context: dg.AssetExecutionContext):
    run_pyrenew_model(context, model_letters="hew", model_family="pyrenew")
    return "pyrenew_hew_output"


# --------------------------------------------------------- #
# Runtime Configuration: Working Directory, Executors
# - Executors define the runtime-location of an asset job
# - See later on for Asset job definitions
# --------------------------------------------------------- #"

workdir = "pyrenew-hew"
local_workdir = Path(__file__).parent.resolve()
image = "cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest"

# add this to a job or the Definitions class to use it
docker_executor_configured = docker_executor.configured(
    {
        # specify a default image
        "image": image,
        "env_vars": [f"DAGSTER_USER={user}", "VIRTUAL_ENV=/pyrenew-hew/.venv"],
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
# Container app jobs cant have mounted volumes, so we would need to refactor pyrenew-hew to use this
azure_caj_executor_configured = azure_caj_executor.configured(
    {
        "image": image,
        "env_vars": [f"DAGSTER_USER={user}", "VIRTUAL_ENV=/pyrenew-hew/.venv"],
    }
)

# configuring an executor to run workflow steps on Azure Batch 4CPU 16GB RAM pool
# add this to a job or the Definitions class to use it
azure_batch_executor_configured = azure_batch_executor.configured(
    {
        "pool_name": "pyrenew-pool",
        "image": image,
        "env_vars": [f"DAGSTER_USER={user}", "VIRTUAL_ENV=/pyrenew-hew/.venv"],
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
            "working_dir": "/pyrenew-hew",
        },
    }
)

# -------------------------------------------------------------------------- #
# Asset Jobs and Schedules: how are outputs created together and when?
# -------------------------------------------------------------------------- #
upstream_asset_job = dg.define_asset_job(
    name="upstream_asset_job",
    executor_def=dg.in_process_executor,
    selection=["nhsn_data", "nssp_gold", "nssp_latest_comprehensive", "nwss_data"],
    # tag the run with your user to allow for easy filtering in the Dagster UI
    tags={"user": user},
)

pyrenew_asset_job = dg.define_asset_job(
    name="pyrenew_asset_job",
    executor_def=azure_batch_executor_configured,
    selection=[
        timeseries_e_output,
        pyrenew_e_output,
        pyrenew_h_output,
        pyrenew_he_output,
        pyrenew_hw_output,
        pyrenew_hew_output,
    ],
    # tag the run with your user to allow for easy filtering in the Dagster UI
    tags={"user": user},
)

upstream_every_wednesday = dg.ScheduleDefinition(
    name="weekly_upstream_cron", cron_schedule="0 9 * * 3", job=upstream_asset_job
)

pyrenew_every_wednesday = dg.ScheduleDefinition(
    name="weekly_pyrenew_cron", cron_schedule="0 9 * * 3", job=pyrenew_asset_job
)

# env variable set by Dagster CLI
is_production = not os.getenv("DAGSTER_IS_DEV_CLI")

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
    resources={
        # This IOManager lets Dagster serialize asset outputs and store them
        # in Azure to pass between assets
        "io_manager": ADLS2PickleIOManager(),
        # an example storage account
        "azure_blob_storage": AzureBlobStorageResource(
            account_url=f"{storage_account}.blob.core.windows.net",
            credential=AzureBlobStorageDefaultCredential(),
        ),
    },
    # setting Docker as the default executor. comment this out to use
    # the default executor that runs directly on your computer
    # executor=docker_executor_configured,
    # executor=dg.in_process_executor,
    # executor=azure_caj_executor_configured,
    executor=azure_batch_executor_configured,
    # uncomment the below to launch runs on Azure CAJ
    metadata={
        "cfa_dagster/launcher": {
            "class": AzureContainerAppJobRunLauncher.__name__,
            "config": {
                "image": image,
            },
        }
    },
)
