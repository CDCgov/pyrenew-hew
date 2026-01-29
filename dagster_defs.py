# Basic Imports
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# Dagster and cloud Imports
import dagster as dg
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from cfa_dagster import (
    ADLS2PickleIOManager,
    AzureContainerAppJobRunLauncher,
    azure_batch_executor,
    collect_definitions,
    docker_executor,
    launch_asset_backfill,
    start_dev_env,
)
from cfa_dagster import azure_container_app_job_executor as azure_caj_executor
from dagster_azure.blob import (
    AzureBlobStorageDefaultCredential,
    AzureBlobStorageResource,
)

# Model Code
# Model Code
# from pipelines.forecast_pyrenew import main as forecast_pyrenew
# from pipelines.forecast_timeseries import main as forecast_timeseries
# from pipelines.postprocess_forecast_batches import main as postprocess
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

# Disease Partitions
disease_partitions = dg.StaticPartitionsDefinition(["COVID-19", "Influenza", "RSV"])

# State Partitions
# fmt: off
state_partitions = dg.StaticPartitionsDefinition([
    'US', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO',
    'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
    'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
    'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
    'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
    'WV', 'WI', 'WY', 'AS', 'GU', 'MP', 'PR',
    'UM', 'VI'
])
# fmt: off

# Multi Partitions
pyrenew_multi_partition_def = dg.MultiPartitionsDefinition({
        "disease": disease_partitions,
          "state": state_partitions
})

# ----------------------------------------------------------- #
# Asset Definitions - What are we outputting in our pipeline?
# ----------------------------------------------------------- #

# ---------------
# Asset Configs
# ---------------
class CommonConfig(dg.Config):
    """
    Common configuration for both Model and Post-Processing assets.
    Both ModelConfig and PostProcessConfig inherit from this, then add their own parameters.
    """

    forecast_date: str = datetime.now(UTC).strftime("%Y-%m-%d")
    # _output_basedir: str = "output" if is_production else "test-output"
    _output_basedir: str = "test-output"
    _output_subdir: str = f"{forecast_date}_forecasts"
    output_dir: str = f"{_output_basedir}/{_output_subdir}"

class ModelConfig(CommonConfig):
    """
    Configuration for the Pyrenew model assets.
    These default values can be modified in the Dagster asset materialization launchpad.
    We also unpack these for our job run configurations.
    """

    # Parameters that are currently defined in asset functions - this may be tweakable later
    # model_letters: str = "hew"  # experimental - not ready to incorporate in config yet
    # model_family: str = "pyrenew"  # experimental - not ready to incorporate in config yet
    # manual_state_exclusions: list[str] = [] # experimental - not ready to incorporate in config yet

    # Parameters that can be toggled in the launchpad or in custom jobs
    # test: bool = True if not is_production else False
    test: bool = True
    n_training_days: int = 150
    exclude_last_n_days: int = 1
    n_warmup: int = 200 if test else 1000
    n_samples: int = 200 if test else 500
    n_chains: int = 2 if test else 4
    n_total_samples: int = n_samples * n_chains
    rng_key: int = 12345
    additional_forecast_letters: str = ""


class PostProcessConfig(CommonConfig):
    """
    Configuration for the Post-Processing asset.
    """
    skip_existing: bool = True
    save_local_copy: bool = False
    local_copy_dir: str = "" # "stf_forecast_fig_share"

# ----------------------
# Model Worker Function
# ----------------------

# This function is NOT an asset itself, but iscalled by assets to run the pyrenew model
def run_pyrenew_model(
    context: dg.AssetExecutionContext,
    config: ModelConfig,
    model_letters: str,
    model_family: str,
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

    # Configuration inherited from ModelConfig
    context.log.debug(f"config: '{config}'")

    # =====================================
    # Model Family and Run Script Selection
    # =====================================
    if model_family == "pyrenew":
        # from forecast_pyrenew import forecast_pyrenew  # noqa: F401
        run_script = "forecast_pyrenew.py"
        additional_args = (
            f"--n-samples {config.n_samples} "
            f"--n-chains {config.n_chains} "
            f"--n-warmup {config.n_warmup} "
            "--nwss-data-dir nwss-vintages "
            "--priors-path pipelines/priors/prod_priors.py "
            f"--rng-key {config.rng_key} "
        )
        if config.additional_forecast_letters:
            additional_args += f"--additional-forecast-letters {config.additional_forecast_letters} "
    elif model_family == "timeseries":
        run_script = "forecast_timeseries.py"
        additional_args = f"--n-samples {config.n_total_samples} "
    else:
        raise ValueError(
            f"Unsupported model family: {model_family}. "
            "Supported values are 'pyrenew' and 'timeseries'."
        )

    # =======================================
    # Azure Batch Script Command Construction
    # =======================================

    # TODO: investigate calling the run_script function directly instead of via shell-nested python subprocess
    base_call = (
        "/bin/bash -c '"
        f"uv run python pipelines/{run_script} "
        f"--disease {disease} "
        f"--loc {state} "
        f"--n-training-days {config.n_training_days} "
        "--facility-level-nssp-data-dir nssp-etl/gold "
        "--param-data-dir params "
        f"--output-dir {config.output_dir} "
        "--credentials-path config/creds.toml "
        f"--exclude-last-n-days {config.exclude_last_n_days} "
        f"--model-letters {model_letters} "
        f"{additional_args}"
        "'"
    )

    subprocess.run(base_call, shell=True, check=True)


# --------------------------------------------------------------------
# Assets: these are the core of Dagster - functions that specify data
# --------------------------------------------------------------------

# -- Pyrenew Assets -- #

# Timeseries E
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
)
def timeseries_e(context: dg.AssetExecutionContext, config: ModelConfig):
    """
    Run Timeseries-e model and produce outputs.
    """
    run_pyrenew_model(context, config, model_letters="e", model_family="timeseries")
    return "timeseries_e"


# Pyrenew E
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
    deps="timeseries_e"
)
def pyrenew_e(context: dg.AssetExecutionContext, config: ModelConfig, timeseries_e):
    """
    Run Pyrenew-e model and produce outputs.
    """
    run_pyrenew_model(context, config, model_letters="e", model_family="pyrenew")
    return "pyrenew_e"


# Pyrenew H
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
)
def pyrenew_h(context: dg.AssetExecutionContext, config: ModelConfig):
    """
    Run Pyrenew-h model and produce outputs.
    """
    run_pyrenew_model(context, config, model_letters="h", model_family="pyrenew")
    return "pyrenew_h"


# Pyrenew HE
@dg.asset(
    partitions_def=pyrenew_multi_partition_def
)
def pyrenew_he(context: dg.AssetExecutionContext, config: ModelConfig, timeseries_e):
    """
    Run Pyrenew-he model and produce outputs.
    """
    run_pyrenew_model(context, config, model_letters="he", model_family="pyrenew")
    return "pyrenew_he"


# Pyrenew HW
@dg.asset(
    partitions_def=pyrenew_multi_partition_def,
)
def pyrenew_hw(context: dg.AssetExecutionContext, config: ModelConfig):
    """
    Run Pyrenew-hw model and produce outputs.
    """
    run_pyrenew_model(context, config, model_letters="hw", model_family="pyrenew")
    return "pyrenew_hw"


# Pyrenew HEW
@dg.asset(
    partitions_def=pyrenew_multi_partition_def
)
def pyrenew_hew(context: dg.AssetExecutionContext, config: ModelConfig, timeseries_e):
    """
    Run Pyrenew-hew model and produce outputs.
    """
    run_pyrenew_model(context, config, model_letters="hew", model_family="pyrenew")
    return "pyrenew_hew"

# Use this template for new models. If needed, add dependencies as an argument and overrides in the config

# @dg.asset(
#     partitions_def=pyrenew_multi_partition_def
# )
# def pyrenew_generic(context: dg.AssetExecutionContext, config: ModelConfig):
#     run_pyrenew_model(context, config, model_letters="<?>", model_family="pyrenew")
#     return "pyrenew_generic"


# -- Postprocessing Forecast Batches -- #

@dg.asset(
    partitions_def=disease_partitions
)
def postprocess_forecasts(
    context: dg.AssetExecutionContext, config: PostProcessConfig,
    timeseries_e, pyrenew_e, pyrenew_h, pyrenew_he, pyrenew_hw, pyrenew_hew
):
    """
    Postprocess forecast batches for a given disease partition.
    """
    disease = context.partition_key
    print(disease)
    # postprocess(
    #     base_forecast_dir=config.output_dir,
    #     diseases=list(disease),
    #     skip_existing=config.skip_existing,
    #     local_copy_dir=config.output_dir,
    # )
    return "postprocess_forecasts"

# --------------------------------------------------------- #
# Runtime Configuration: Working Directory, Executors
# - Executors define the runtime-location of an asset job
# - See later on for Asset job definitions
# --------------------------------------------------------- #"

workdir = "pyrenew-hew"
local_workdir = Path(__file__).parent.resolve()
image = "cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest"

# add this to a job or the Definitions class to use it
docker_executor_configured = docker_executor.configured({
        "image": image,
        "env_vars": [f"DAGSTER_USER={user}", "VIRTUAL_ENV=/pyrenew-hew/.venv"],
        "retries": {"enabled": {}},
        "container_kwargs": {
            "volumes": [
                # bind the ~/.azure folder for optional cli login
                f"/home/{user}/.azure:/root/.azure",
                # bind current file so we don't have to rebuild
                # the container image for workflow changes
                f"{__file__}:/{workdir}/{os.path.basename(__file__)}",
                # blob container mounts for pyrenew-hew
                f"/{local_workdir}/nssp-archival-vintages:/pyrenew-hew/nssp-archival-vintages",
                f"/{local_workdir}/nssp-etl:/pyrenew-hew/nssp-etl",
                f"/{local_workdir}/nwss-vintages:/pyrenew-hew/nwss-vintages",
                f"/{local_workdir}/params:/pyrenew-hew/params",
                f"/{local_workdir}/config:/pyrenew-hew/config",
                f"/{local_workdir}/output:/pyrenew-hew/output",
                f"/{local_workdir}/test-output:/pyrenew-hew/test-output",
            ]
        },
})

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
        "pool_name": "pyrenew-dagster-pool",
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

# Azure Container App Job Launcher
azure_caj_launcher = {
    "cfa_dagster/launcher": {
        "class": AzureContainerAppJobRunLauncher.__name__,
        "config": {
            "image": image,
        },
    }
}

# Standard Docker - uses the default run launcher
docker_metadata = {"executor": docker_executor_configured}

# For use with Container App Jobs
azure_caj_metadata = {
    "executor": azure_caj_executor_configured,
    "metadata": azure_caj_launcher,
}

# For user with Azure Batch Jobs
azure_batch_metadata = {
    "executor": azure_batch_executor_configured,
    "metadata": azure_caj_launcher if not is_production else {},
}

# ------------------------------------------------------------------------------------------------- #
# Orchestration of Partitioned Assets - Model Runs
#
# Scheduling full pipeline runs and defining a flexible configuration
# We use dagster ops and jobs here to launch asset backfills with custom configuration
# ------------------------------------------------------------------------------------------------- #

## Prototype - Simple Asset Job Definition and Schedule ##

# Experimental asset job to materialize all pyrenew assets
run_pyrenew_hew_gui = dg.define_asset_job(
    name="RunPyrenewHew_GUI",
    executor_def=azure_batch_executor_configured,
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

run_postprocess_forecasts_gui = dg.define_asset_job(
    name="RunPostprocessForecasts_GUI",
    executor_def=dg.multiprocess_executor if not is_production else azure_batch_executor_configured,
    selection=[postprocess_forecasts],
    tags={"user": user},
)

## Data Availability Check Functions ##

@dg.op
def check_nhsn_data_availability():
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    nhsn_target_url = "https://data.cdc.gov/api/views/mpgq-jmmr.json"
    try:
        resp = requests.get(nhsn_target_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        nhsn_update_date_raw = data.get("rowsUpdatedAt")
        if nhsn_update_date_raw is None:
            return {"exists": False, "reason": "Key 'rowsUpdatedAt' not found"}
        nhsn_update_date = datetime.utcfromtimestamp(nhsn_update_date_raw).strftime("%Y-%m-%d")
        nhsn_check = nhsn_update_date == current_date
        print(f"NHSN data available for date {current_date}: {nhsn_check}")
        return {
            "exists": nhsn_check,
            "update_date": nhsn_update_date,
            "current_date": current_date,
        }
    except Exception as e:
        print(f"Error checking NHSN data availability: {e}")
        return {"exists": False, "reason": str(e)}

@dg.op
def check_nssp_gold_data_availability(account_name="cfaazurebatchprd", container_name="nssp-etl"):
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    blob_name = f"gold/{current_date}.parquet"
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        f"https://{account_name}.blob.core.windows.net", credential=credential
    )
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs(name_starts_with=blob_name))
    nssp_gold_check = bool(blobs)
    latest_blob = None
    blobs_gold = list(container_client.list_blobs(name_starts_with="gold/"))
    if blobs_gold:
        latest_blob = max(blobs_gold, key=lambda b: b.last_modified).name
    print(f"NSSP gold data avaialble for date {current_date}: {nssp_gold_check}")
    return {
        "exists": nssp_gold_check,
        "blob_name": blob_name,
        "latest_blob": latest_blob,
        "current_date": current_date,
    }

@dg.op
def check_nwss_gold_data_availability(account_name="cfaazurebatchprd", container_name="nwss-vintages"):
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    folder_prefix = f"NWSS-ETL-covid-{current_date}/"
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        f"https://{account_name}.blob.core.windows.net", credential=credential
    )
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs(name_starts_with=folder_prefix))
    nwss_gold_check = bool(blobs)
    print(f"NWSS gold data avaialble for date {current_date}: {nwss_gold_check}")
    return {
        "exists": nwss_gold_check,
        "folder_prefix": folder_prefix,
        "current_date": current_date,
    }

## Backfill Launch Method - Flexible Configuration via Op and Job ##

# This is an op (non-materialized asset function) that launches backfills, as used in scheduled jobs
@dg.op
def launch_pyrenew_pipeline(
    context: dg.OpExecutionContext,
    config: ModelConfig
) -> dg.Output[str]:

    # We are referencing the global pyrenew_multi_partition_def defined earlier
    partition_keys = pyrenew_multi_partition_def.get_partition_keys() # TODO: remove slice and/or config-parametrize  once ready for prod

    asset_selection = ()

    # Determine which assets to backfill based on data availability
    nhsn_available = check_nhsn_data_availability()["exists"] # H Data
    nssp_available = check_nssp_gold_data_availability()["exists"] # E Data
    # nwss_available = check_nwss_gold_data_availability()["exists"] # W Data

    context.log.debug(f"NHSN available: {nhsn_available}")
    context.log.debug(f"NSSP available: {nssp_available}")
    # context.log.debug(f"NWSS available: {nwss_available}")

    # Determine which assets to backfill based on data availability
    # if nhsn_available and nssp_available and nwss_available:
    #     context.log.info("NHSN, NSSP gold, and NWSS gold data are all available - launching full pipeline.")
    #     context.log.info("Launching full pyrenew_hew backfill.")
    #     asset_selection = ("timeseries_e", "pyrenew_e", "pyrenew_h", "pyrenew_he", "pyrenew_hw", "pyrenew_hew")

    if nhsn_available and nssp_available:
    # elif nhsn_available and nssp_available:
        context.log.info("Both NHSN data and NSSP gold data are available, but NWSS gold data is not.")
        context.log.info("Launching a timeseries_e, pyrenew_e, pyrenew_h, and pyrenew_he backfill.")
        asset_selection = ("timeseries_e", "pyrenew_e", "pyrenew_h", "pyrenew_he")

    # elif nhsn_available and nwss_available:
    #     context.log.info("NHSN data and NWSS data are available, but NSSP gold data is not.")
    #     context.log.info("Launching pyrenew_h and pyrenew_hw backfill.")
    #     asset_selection = ("pyrenew_h", "pyrenew_hw")

    elif nssp_available:
        context.log.info("Only NSSP gold data are available.")
        context.log.info("Launching a timeseries_e and pyrenew_e backfill.")
        asset_selection = ("timeseries_e", "pyrenew_e")

    elif nhsn_available:
        context.log.info("Only NHSN data are available.")
        context.log.info("Launching a pyrenew_h backfill.")
        asset_selection = ("pyrenew_h")

    else:
        context.log.info("No required data is available.")
        asset_selection = ()


    # Launch the backfill
    # Returns: a backfill ID,
    # side-effect: launches the backfill run in Dagster via a GraphQL query
    backfill_id = launch_asset_backfill(
        asset_selection,
        partition_keys,
        run_config=dg.RunConfig({
                "ops": {
                    **{asset: config for asset in asset_selection},
                },
        }),
        tags={
                "run": "pyrenew",
                "available_data": str({
                    "nhsn": nhsn_available,
                    "nssp_gold" : nssp_available,
                    "nwss_gold" : False, # nwss_available,
                }),
                "user": user,
                "models_attempted": ", ".join(asset_selection),
                "forecast_date": config.forecast_date,
                "output_dir": config.output_dir,
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
    },
    config=dg.RunConfig(
        ops={
            "launch_pyrenew_pipeline": ModelConfig()
        }
    )
)
def weekly_pyrenew_via_backfill():
    launch_pyrenew_pipeline()


weekly_pyrenew_via_backfill_schedule = dg.ScheduleDefinition(
    default_status=(
        dg.DefaultScheduleStatus.RUNNING
        # don't run locally by default
        if is_production else dg.DefaultScheduleStatus.STOPPED
    ),
    job=weekly_pyrenew_via_backfill,
    run_config=dg.RunConfig(
        ops={
            "launch_pyrenew_pipeline": ModelConfig()
        }
    ),
    cron_schedule="0 15 * * WED",
    execution_timezone="America/New_York",
)

@dg.job(
    executor_def=dg.multiprocess_executor,
    tags={
        "cfa_dagster/launcher": {
            "class": dg.DefaultRunLauncher.__name__
        }
    }
)
def check_all_data():
    nhsn=check_nhsn_data_availability() # H Data
    print("NHSN data availability", nhsn)
    nssp=check_nssp_gold_data_availability() # E Data
    print("NSSP gold data availability", nssp)
    nwss=check_nwss_gold_data_availability() # W Data
    print("NWSS gold data availability", nwss)

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
    # New ** syntax combines executor and launcher metadata
    **(
        azure_batch_metadata # comment when testing locally, take care not to submit too many jobs
        # docker_metadata # use this when running locally - be careful not to use the backfill job locally
    ),
)
