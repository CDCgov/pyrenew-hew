import os
import subprocess
from pathlib import Path

import dagster as dg
from cfa_dagster.azure_container_app_job.executor import (
    azure_container_app_job_executor,
)
from cfa_dagster.docker.executor import docker_executor

# end_offset=1 allows you to target the current month
# by default, dagster only allows you to materialize after a month is complete
monthly_partition = dg.MonthlyPartitionsDefinition(
    start_date="2024-01-01", end_offset=1
)

# get the user from the environment, throw an error if variable is not set
user = os.environ["DAGSTER_USER"]


# class SomeRAssetConfig(dg.Config):
#     # when using the docker_executor, specify the image you'd like to use
#     image: str = f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}"

# class SomePythonAssetConfig(dg.Config):
#     # when using the docker_executor, specify the image you'd like to use
#     image: str = f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}"

class UpstreamAssetConfig(dg.Config):
    # when using the docker_executor, specify the image you'd like to use
    image: str = f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}"

class PyrenewAssetConfig(dg.Config):
    # when using the docker_executor, specify the image you'd like to use
    image: str = f"cfaprdbatchcr.azurecr.io/cfa-dagster-sandbox:{user}"

# @dg.asset(
#     automation_condition=dg.AutomationCondition.eager(),
#     description="An asset that runs R code",
# )
# def some_r_asset(context, config: SomeRAssetConfig):
#     run = subprocess.run("Rscript hello.R", shell=True)
#     run.check_returncode()

# @dg.asset(
#     automation_condition=dg.AutomationCondition.eager(),
#     description="An asset that runs Python code",
# )
# def some_python_asset(context, config: SomePythonAssetConfig):
#     run = subprocess.run("uv run hello.py", shell=True)
#     run.check_returncode()

# Upstream Assets
@dg.asset
def nssp_gold() -> str:
    # this shoudld get the nssp gold data
    return "nssp-gold"

@dg.asset(
        deps=['nssp_gold']
)
def nssp_latest_comprehensive() -> str:
    # this should pull the nssp latest comprehensive data
    return "nssp-latest-comprehensive"

@dg.asset
def nwss_gold() -> str:
    # this should get the nwss gold data
    return "nwss-gold"

@dg.asset
def nhsn_latest() -> str:
    # this should pull the nhsn data
    return "nhsn-latest"

# Pyrenew Assets
@dg.asset
def timeseries_e_output(nssp_gold, nssp_latest_comprehensive) -> str:
    # These should generate the outputs by submitting to azure batch.
    return "nssp-timeseries-e-output"

@dg.asset(
    deps=['timeseries_e_output', 'nssp_latest_comprehensive']
)
def pyrenew_e_output() -> str:
    # These should generate the outputs by submitting to azure batch.
    return "pyrenew-e-output"

@dg.asset(
    deps=['nhsn_latest', 'nssp_gold', 'timeseries_e_output', 'nssp_latest_comprehensive']
)
def pyrenew_he_output() -> str:
    # These should generate the outputs by submitting to azure batch.
    return "pyrenew-he-output"

@dg.asset(
    deps=['nhsn_latest', 'nwss_gold', 'nssp_gold', 'timeseries_e_output', 'nssp_latest_comprehensive']
)
def pyrenew_hew_output() -> str:
    # These should generate the outputs by submitting to azure batch.
    return "pyrenew-hew-output"

disease_partitions = dg.StaticPartitionsDefinition(["COVID-19", "INFLUENZA", "RSV"])
state_partitions = dg.StaticPartitionsDefinition(["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA"])
two_dimensional_partitions = dg.MultiPartitionsDefinition(
    {"disease": disease_partitions, "loc": state_partitions}
)

class PyrenewHOutputConfig(dg.Config):
    # when using the docker_executor, specify the image you'd like to use
    image: str = f"pyrenew-dagster"

@dg.asset(
    partitions_def=two_dimensional_partitions,
)
def pyrenew_h_output(context: dg.AssetExecutionContext, config: PyrenewHOutputConfig, nhsn_latest) -> str:
    # These should generate the outputs by submitting to azure batch.
    # Trace down all the variables.
    keys_by_dimension: dg.MultiPartitionKey = context.partition_key.keys_by_dimension
    disease = keys_by_dimension["disease"]
    loc = keys_by_dimension["loc"]
    run_script = "forecast_pyrenew.py"
    n_training_days = 150
    n_samples = 500
    exclude_last_n_days = 1
    model_letters = "h"
    n_warmup = 1000
    additional_forecast_letters = ""
    output_subdir = "./"
    additional_args = (
            f"--n-warmup {n_warmup} "
            "--nwss-data-dir nwss-vintages "
            "--priors-path pipelines/priors/prod_priors.py "
            f"--additional-forecast-letters {additional_forecast_letters} "
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
    base_call=base_call.format(
                loc=loc,
                disease=disease,
                report_date="latest",
                output_dir=str(Path("output", output_subdir)),
    )
    run = subprocess.run(base_call, shell=True, check=True)
    run.check_returncode()
    return "pyrenew-h-output"

@dg.asset(
    deps=['nhsn_latest','nwss_gold']
)
def pyrenew_hw_output() -> str:
    # These should generate the outputs by submitting to azure batch.
    return "pyrenew-hw-output"


# path to src folder for docker volume binding
src_path = Path(__file__).resolve().parent.parent.parent

upstream_assets_job = dg.define_asset_job(
    name="upstream_assets_job",
    executor_def=docker_executor,
    # uncomment the below to switch to run on Azure
    # executor_def=azure_container_app_job_executor,
    selection=["nssp_gold", "nssp_latest_comprehensive", "nwss_gold", "nhsn_latest"],
    config=dg.RunConfig(
        execution={
            "config": {
                "env_vars": [f"DAGSTER_USER={user}"],
                "container_kwargs": {
                    "volumes": [
                        # bind the ~/.azure folder for cli login
                        f"/home/{user}/.azure:/root/.azure",
                        # bind src path so we don't have to rebuild
                        # the container image for workflow changes
                        f"{src_path}:/app/src",
                    ]
                },
            }
        },
        ops={
            # tell the job to use your asset config which includes your image
            "nssp_gold": UpstreamAssetConfig(),
            "nssp_latest_comprehensive": UpstreamAssetConfig(),
            "nwss_gold": UpstreamAssetConfig(),
            "nhsn_latest": UpstreamAssetConfig(),
        },
    ),
)

pyrenew_assets_job = dg.define_asset_job(
    name="pyrenew_assets_job",
    executor_def=docker_executor,
    # uncomment the below to switch to run on Azure
    # executor_def=azure_container_app_job_executor,
    selection=["pyrenew_h_output"],
    # selection=["timeseries_e_output", "pyrenew_e_output", "pyrenew_h_output", "pyrenew_he_output", "pyrenew_hw_output", "pyrenew_hew_output"],
    config=dg.RunConfig(
        execution={
            "config": {
                "env_vars": [f"DAGSTER_USER={user}"],
                "container_kwargs": {
                    "volumes": [
                        # bind the ~/.azure folder for cli login
                        f"/home/{user}/.azure:/root/.azure",
                        # bind src path so we don't have to rebuild
                        # the container image for workflow changes
                        f"{src_path}:/app/src",
                    ]
                },
            }
        },
        ops={
            # tell the job to use your asset config which includes your image
            "pyrenew_h_output": PyrenewHOutputConfig(),
        },
    ),
)

# some_asset_job = dg.define_asset_job(
#     name="some_asset_job",
#     executor_def=docker_executor,
#     # uncomment the below to switch to run on Azure
#     # executor_def=azure_container_app_job_executor,
#     selection=["some_r_asset", "some_python_asset"],
#     config=dg.RunConfig(
#         execution={
#             "config": {
#                 "env_vars": [f"DAGSTER_USER={user}"],
#                 "container_kwargs": {
#                     "volumes": [
#                         # bind the ~/.azure folder for cli login
#                         f"/home/{user}/.azure:/root/.azure",
#                         # bind src path so we don't have to rebuild
#                         # the container image for workflow changes
#                         f"{src_path}:/app/src",
#                     ]
#                 },
#             }
#         },
#         ops={
#             # tell the job to use your asset config which includes your image
#             "some_r_asset": SomeRAssetConfig(),
#             "some_python_asset": SomePythonAssetConfig(),
#         },
#     ),
# )

