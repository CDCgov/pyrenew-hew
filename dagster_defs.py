import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
from cfa_dagster import (
    ADLS2PickleIOManager,
    azure_batch_executor,
    collect_definitions,
    docker_executor,
    launch_asset_backfill,
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

# env variable set by Dagster CLI
is_production = not os.getenv("DAGSTER_IS_DEV_CLI")

# get the user running the Dagster instance
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

disease_list = [
    "COVID-19", "Influenza", "RSV"
]

# Disease Partitions
disease_partitions = dg.StaticPartitionsDefinition(
    disease_list
)

# State Partitions
state_partitions = dg.StaticPartitionsDefinition(
    full_state_list
)

# Multi Partitions
pyrenew_multi_partition_def = dg.MultiPartitionsDefinition(
    {
        "disease": disease_partitions,
          "state": state_partitions
    }
)

# ----------------------------------------------------------- #
# Asset Definitions - What are we outputting in our pipeline?
# ----------------------------------------------------------- #

# ---------------
# Asset Configs
# ---------------

class PyrenewAssetConfig(dg.Config):
    """
    Configuration for the Pyrenew model assets.
    These default values can be modified in the Dagster asset materialization launchpad.
    We also unpack these for our job run configurations.
    """
    n_training_days: int = 150
    n_samples: int = 500
    exclude_last_n_days: int = 1
    n_warmup: int = 1000
    additional_forecast_letters: str = ""
    forecast_date: str = datetime.now(UTC).strftime("%Y-%m-%d")
    output_dir: str = "output" if is_production else "test-output"
    output_subdir: str = f"{forecast_date}_forecasts"
    full_dir: str = f"{output_dir}/{output_subdir}"

# ---------------
# Worker Function
# ---------------

# This function is NOT an asset itself, but is called by assets to run the pyrenew model
def run_pyrenew_model(
    context: dg.AssetExecutionContext,
    config: PyrenewAssetConfig,
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

    # Configuration inherited from PyrenewAssetConfig
    context.log.debug(f"config: '{config}'")
    n_training_days = config.n_training_days
    n_samples = config.n_samples
    exclude_last_n_days = config.exclude_last_n_days
    n_warmup = config.n_warmup
    additional_forecast_letters = config.additional_forecast_letters
    full_dir = config.full_dir

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
        f"--output-dir {full_dir} "
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

# -- Upstream Data -- #

# NOTE: These are placeholder assets representing data ingestion steps.
# In mature productionimplementation, these would contain logic to ingest and process data,
# or sensors to trigger on new data availability.

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


# Dependency Definitions #

nssp_deps = ["nssp_gold", "nssp_latest_comprehensive"]

# -- Pyrenew Assets -- #

# TODO: adapt materialize results for asset returns. Currently returning simple strings.
# i.e.
# return dg.MaterializeResult(
#         value=output_path,
#         metadata={
#             "config": config,
#             "output_path": output_path,
#             "storage_account": STORAGE_ACCOUNT,
#             "storage_container": OUTPUT_CONTAINER,
#             "blob_path": job_id,
#         }
#     )


# Timeseries E
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
    deps=nssp_deps,
)
def timeseries_e(context: dg.AssetExecutionContext, config: PyrenewAssetConfig):
    run_pyrenew_model(context, config, model_letters="e", model_family="timeseries")
    return "timeseries_e"


# Pyrenew E
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
    deps=["timeseries_e"] + nssp_deps,
)
def pyrenew_e(context: dg.AssetExecutionContext, config: PyrenewAssetConfig):
    run_pyrenew_model(context, config, model_letters="e", model_family="pyrenew")
    return "pyrenew_e"


# Pyrenew H
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
    deps=["nhsn_data"],
)
def pyrenew_h(context: dg.AssetExecutionContext, config: PyrenewAssetConfig):
    run_pyrenew_model(context, config, model_letters="h", model_family="pyrenew")
    return "pyrenew_h"


# Pyrenew HE
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
    deps=["timeseries_e", "nhsn_data"] + nssp_deps,
)
def pyrenew_he(context: dg.AssetExecutionContext, config: PyrenewAssetConfig):
    run_pyrenew_model(context, config, model_letters="he", model_family="pyrenew")
    return "pyrenew_he"


# Pyrenew HW
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
    deps=["nhsn_data", "nwss_data"],
)
def pyrenew_hw(context: dg.AssetExecutionContext, config: PyrenewAssetConfig):
    run_pyrenew_model(context, config, model_letters="hw", model_family="pyrenew")
    return "pyrenew_hw"


# Pyrenew HEW
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
    deps=["timeseries_e"] + nssp_deps + ["nhsn_data", "nwss_data"],
)
def pyrenew_hew(context: dg.AssetExecutionContext, config: PyrenewAssetConfig):
    run_pyrenew_model(context, config, model_letters="hew", model_family="pyrenew")
    return "pyrenew_hew"


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

# ----------------------------------------------------------------------------------- #
# Orchestration of Non-Partitioned Assets - Upstream Data ETL
#
# Note that this basic approach only works for non-partitioned assets,
# the models we'll need to schedule more complexly
# ----------------------------------------------------------------------------------- #

# TODO: This will be replaced by sensors in production
upstream_asset_job = dg.define_asset_job(
    name="upstream_asset_job",
    executor_def=dg.in_process_executor, # these are lightweight and do not have partitions
    selection=["nhsn_data", "nssp_gold", "nssp_latest_comprehensive", "nwss_data"],
    # tag the run with your user to allow for easy filtering in the Dagster UI
    tags={"user": user},
)

# Every wednesday this will run at hours 11 through 20 UTC (6am-3pm EST)
# upstream_every_wednesday = dg.ScheduleDefinition(
#     name="weekly_upstream_cron", cron_schedule="0 11-20 * * WED", job=upstream_asset_job
# )

# ------------------------------------------------------------------------------------------------- #
# Orchestration of Partitioned Assets - Model Runs
#
# Scheduling full pipeline runs and defining a flexible configuration
# We use dagster ops and jobs here to launch asset backfills with custom configuration
# ------------------------------------------------------------------------------------------------- #

## Prototype - Simple Asset Job Definition and Schedule ##

# Experimental asset job to materialize all pyrenew assets
naive_pyrenew_asset_job = dg.define_asset_job(
    name="naive_pyrenew_asset_job",
    executor_def=azure_batch_executor_configured, # these are lightweight and do not have partitions
    selection=[
        "timeseries_e",
        "pyrenew_e",
        "pyrenew_h",
        "pyrenew_he",
        "pyrenew_hw",
        "pyrenew_hew",
    ],
    # tag the run with your user to allow for easy filtering in the Dagster UI
    tags={"user": user},
)

# naive_pyrenew_test_schedule = dg.ScheduleDefinition(
#     default_status=(
#         dg.DefaultScheduleStatus.RUNNING
#         # don't run locally by default
#         if is_production else dg.DefaultScheduleStatus.STOPPED
#     ),
#     job=naive_pyrenew_asset_job,
#     cron_schedule="0 12-21 * * WED",
#     execution_timezone="America/New_York",
# )

## Backfill Launch Method - Flexible Configuration via Op and Job ##

# This is an op (non-materialized asset function) that launches backfills, as used in scheduled jobs
@dg.op
def launch_pyrenew_pipeline(
    context: dg.OpExecutionContext,
    config: PyrenewAssetConfig,
) -> dg.Output[str]:

    # We are referencing the global pyrenew_multi_partition_def defined earlier
    partition_keys = pyrenew_multi_partition_def.get_partition_keys()[3:23]

    # We select all the assets we want to "backfill"
    asset_selection = (
        "timeseries_e",
        "pyrenew_e",
        "pyrenew_h",
        "pyrenew_he",
        # "pyrenew_hw",
        # "pyrenew_hew",
    )

    # Launch the backfill
    # Returns: a backfill ID,
    # side-effect: launches the backfill run in Dagster via a GraphQL query
    backfill_id = launch_asset_backfill(
        asset_selection,
        partition_keys,
        run_config=dg.RunConfig({
                "ops": {
                    **{asset: config for asset in asset_selection},
                }
        }),
        tags={
                "run": "pyrenew",
        }
    )
    context.log.info(
        f"Launched backfill with id: '{backfill_id}'. "
        "Click the output metadata url to monitor"
    )
    return dg.Output(
        value=backfill_id,
        metadata={
            "url": dg.MetadataValue.url(f"/runs/b/{backfill_id}")
        }
    )

# This wraps our launch_pipeline op in a job that can be scheduled or manually launched via the GUI
@dg.job(
    executor_def=dg.multiprocess_executor,
    tags={
        "cfa_dagster/launcher": {
            "class": dg.DefaultRunLauncher.__name__
        }
    }
)
def weekly_pyrenew_via_backfill():
    launch_pyrenew_pipeline()


schedule_weekly_pyrenew_via_backfill = dg.ScheduleDefinition(
    default_status=(
        dg.DefaultScheduleStatus.RUNNING
        # don't run locally by default
        if is_production else dg.DefaultScheduleStatus.STOPPED
    ),
    job=weekly_pyrenew_via_backfill,
    run_config=dg.RunConfig(
        ops={
            "launch_pyrenew_pipeline": PyrenewAssetConfig()
        }
    ),
    cron_schedule="0 12-21 * * WED",
    execution_timezone="America/New_York",
)

## Dagster Tutorial Method - Use @dg.schedule ##

# CONTINENTS = [
#     "Africa",
#     "Antarctica",
#     "Asia",
#     "Europe",
#     "North America",
#     "Oceania",
#     "South America",
# ]


# @dg.static_partitioned_config(partition_keys=CONTINENTS)
# def continent_config(partition_key: str):
#     return {"ops": {"continents": {"config": {"continent_name": partition_key}}}}

# class ContinentOpConfig(dg.Config):
#     continent_name: str = "Oceania"

# @dg.asset
# def continents(context: dg.AssetExecutionContext, config: ContinentOpConfig):
#     context.log.info(config.continent_name)


# continent_job = dg.define_asset_job(
#     name="continent_job", selection=[continents], config=continent_config
# )

# @dg.schedule(cron_schedule="0 0 * * *", job=continent_job)
# def continent_schedule():
#     for c in CONTINENTS:
#         yield dg.RunRequest(run_key=c, partition_key=c)


# -------------- Dagster Definitions Object --------------- #
# This code allows us to collect all of the above definitions
# into a single Definitions object for Dagster to read!
# By doing this, we can keep our Dagster code in a single file
# instead of splitting it across multiple files.
# --------------------------------------------------------- #

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
    # metadata={
    #     "cfa_dagster/launcher": {
    #         "class": AzureContainerAppJobRunLauncher.__name__,
    #         "config": {
    #             "image": image,
    #         },
    #     }
    # },
)
