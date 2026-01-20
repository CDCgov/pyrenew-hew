"""
Set up a multi-location, multi-disease run
of pyrenew-hew on Azure Batch.
"""

# Basic Libraries
import argparse
import itertools
from pathlib import Path

# Azure
from azure.batch import models

# Custom CFA Azure Libraries
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job
from azuretools.task import get_container_settings, get_task_config
from forecasttools import location_table

# Rich printing
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
    container_registry: str = "ghcr.io/cdcgov",
    container_image_name: str = "pyrenew-hew",
    container_image_version: str = "latest",
    n_training_days: int = 150,
    exclude_last_n_days: int = 1,
    rng_key: int = 12345,
    locations_include: list[str] | None = None,
    locations_exclude: list[str] | None = None,
    test: bool = False,
    dry_run: bool = False,
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

    rng_key
        Random number generator seed for reproducibility.
        Default 12345.

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

    model_family
        The model family to use for the job. Default 'pyrenew'.
        Supported values are 'pyrenew' and 'timeseries'.

    test
        Is this a testing run? Default ``False``.
        If set, output to pyrenew-test-output.
        If not set, output to pyrenew-hew-prod-output.

    dry_run
        If set, do not submit tasks to Azure Batch.
        Only print what would be done. Default ``False``.

    Returns
    -------
    None
    """

    # ==================
    # Disease Validation
    # ==================
    supported_diseases = ["COVID-19", "Influenza", "RSV"]

    disease_list = diseases

    invalid_diseases = set(disease_list) - set(supported_diseases)
    if invalid_diseases:
        raise ValueError(
            f"Unsupported diseases: {', '.join(invalid_diseases)}; "
            f"supported diseases are: {', '.join(supported_diseases)}"
        )

    # ========================
    # Model Letters Validation
    # ========================
    validate_hew_letters(model_letters)
    additional_forecast_letters = additional_forecast_letters or model_letters
    validate_hew_letters(additional_forecast_letters)

    # =======================
    # Model Family Validation
    # =======================
    if model_family == "timeseries" and model_letters != "e":
        raise ValueError(
            "Only model_letters 'e' is supported for the 'timeseries' model_family."
        )

    # ========================================
    # Output Container and Sampling Parameters
    # ========================================
    pyrenew_hew_output_container = (
        "pyrenew-test-output" if test else "pyrenew-hew-prod-output"
    )
    n_warmup = 200 if test else 1000
    n_samples = 200 if test else 500
    n_chains = 2 if test else 4
    n_total_samples = n_samples * n_chains

    # ==============
    # Location Setup
    # ==============
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

    # ===============
    # Container Setup
    # ===============
    container_image_full_string = (
        f"{container_registry}/{container_image_name}:{container_image_version}"
    )
    container_settings = get_container_settings(
        container_image_full_string,
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

    # =====================================
    # Model Family and Run Script Selection
    # =====================================
    if model_family == "pyrenew":
        run_script = "forecast_pyrenew.py"
        additional_args = (
            f"--n-samples {n_samples} "
            f"--n-chains {n_chains} "
            f"--n-warmup {n_warmup} "
            "--nwss-data-dir nwss-vintages "
            "--priors-path pipelines/priors/prod_priors.py "
            f"--additional-forecast-letters {additional_forecast_letters} "
            f"--rng-key {rng_key} "
        )
    elif model_family == "timeseries":
        run_script = "forecast_timeseries.py"
        additional_args = f"--n-samples {n_total_samples} "
    else:
        raise ValueError(
            f"Unsupported model family: {model_family}. "
            "Supported values are 'pyrenew' and 'timeseries'."
        )

    # =======================================
    # Azure Batch Script Command Construction
    # =======================================
    base_call = (
        "/bin/bash -c '"
        f"uv run python pipelines/{run_script} "
        "--disease {disease} "
        "--loc {loc} "
        f"--n-training-days {n_training_days} "
        "--facility-level-nssp-data-dir nssp-etl/gold "
        "--param-data-dir params "
        "--output-dir {output_dir} "
        "--credentials-path config/creds.toml "
        f"--exclude-last-n-days {exclude_last_n_days} "
        f"--model-letters {model_letters} "
        f"{additional_args}"
        "'"
    )

    # =================
    # Print Job Details
    # =================
    console = Console()

    # Header panel
    console.print(
        Panel.fit(
            "[bold magenta]🚀 pyrenew-hew: Azure Batch Job Submission 🚀",
            border_style="magenta",
        )
    )

    # Job details table
    table = Table(show_header=False, pad_edge=False)
    table.add_row("Job ID", str(job_id))
    table.add_row("Pool ID", str(pool_id))
    table.add_row("Model Family", str(model_family))
    table.add_row("Model Letters", str(model_letters))
    table.add_row("Additional Forecast Letters", str(additional_forecast_letters))
    table.add_row("Diseases", ", ".join(disease_list))
    table.add_row("Output Storage Container", str(pyrenew_hew_output_container))
    table.add_row("Output Subdirectory", str(output_subdir))
    table.add_row("Container Image Full String", str(container_image_full_string))
    table.add_row("Training Days", str(n_training_days))
    table.add_row("Exclude Last N Days", str(exclude_last_n_days))
    table.add_row("RNG Key", str(rng_key))

    # Locations included (5 per line)
    loc_lines = [
        ", ".join(all_locations[i : i + 5]) for i in range(0, len(all_locations), 5)
    ]
    table.add_row("Locations Included", loc_lines[0] if loc_lines else "")
    for loc_line in loc_lines[1:]:
        table.add_row("", loc_line)

    # Excluded locations
    table.add_row("Excluded Locations", ", ".join(all_exclusions))

    def style_bool(val):
        return "[bold green]True[/bold green]" if val else "[grey50]False[/grey50]"

    table.add_row("Test Mode", style_bool(test))
    table.add_row("Dry Run", style_bool(dry_run))

    console.print(table)

    # ================
    # Dry Run Handling
    # ================
    if dry_run:
        console.print("Dry run mode enabled. No tasks will be submitted.")
        console.print("Closing...")
        return None

    # ==========================
    # Azure Batch Job Submission
    # ==========================
    # TODO: Use VM managed identity (or Federated Identity in GH Actions) with DefaultAzureCredential(), output the BatchServiceClient with the Azure SDK instead.
    # We can do an error handling step explicitly defined here to tell the users if their environment needs to be added to an RBAC group or federated identity whitelist.

    print("")
    print("Using environment credentials to authenticate with Azure Batch...")
    creds = EnvCredentialHandler()  # TODO: Jon note: azuretools class. I would love to use DefaultAzureCredential() instead with OIDC and VM identities.
    client = get_batch_service_client(
        creds
    )  # TODO: Jon note: azuretools... but outputs an azure SDK object. IF we can get here with DefaultAzureCredential, it will be more portable.
    print("Submitting job to Azure Batch...")
    job = models.JobAddParameter(
        id=job_id,
        pool_info=models.PoolInformation(pool_id=pool_id),
    )
    create_job(client, job)

    # ===========================
    # Azure Batch Task Submission
    # ===========================
    for disease, loc in itertools.product(disease_list, all_locations):
        task = get_task_config(
            f"{job_id}-{loc}-{disease}-prod",
            base_call=base_call.format(
                loc=loc,
                disease=disease,
                output_dir=str(Path("output", output_subdir)),
            ),
            container_settings=container_settings,
            log_blob_container="pyrenew-hew-logs",
            log_blob_account=creds.azure_blob_storage_account,
            log_subdir=job_id,
            log_compute_node_identity_reference=(creds.compute_node_identity_reference),
        )
        client.task.add(job_id, task)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-letters",
        type=str,
        help=(
            "Fit the model corresponding to the provided model letters (e.g. 'he', 'e', 'hew')."
        ),
    )
    parser.add_argument("--job-id", type=str, help="Name for the Azure batch job")
    parser.add_argument(
        "--pool-id",
        type=str,
        help=("Name of the Azure batch pool on which to run the job"),
        default="pyrenew-pool",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        default="COVID-19 Influenza RSV",
        help=(
            "Name(s) of disease(s) to run as part of the job, "
            "as a whitespace-separated string. Supported "
            "values are 'COVID-19', 'Influenza', and 'RSV'. "
            "Default 'COVID-19 Influenza RSV' (i.e. run for "
            "all three)."
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
        "--container-registry",
        type=str,
        help=("Container registry URL (e.g. ghcr.io/cdcgov) to use for the job."),
        default="ghcr.io/cdcgov",
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
        help=("Number of 'training days' of observed data to use for model fitting."),
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
        "--rng-key",
        type=int,
        help=("Random number generator seed for reproducibility (default: 12345)."),
        default=12345,
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

    # Function to convert string to boolean
    # This is used to allow passing boolean values as command line arguments
    # Reference: https://docs.python.org/3/library/argparse.html#type
    # and
    # https://stackoverflow.com/a/43357954
    def string_to_boolean(value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        if value.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif value.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # With the string_to_boolean argparse type arg,
    # you can supply the flag for True, omit it for False, or pass True/False explicitly
    parser.add_argument(
        "--test",
        type=string_to_boolean,
        nargs="?",
        const=True,
        default=False,
        help="Run in test mode (default: False). Pass --test True or --test False to set explicitly.",
    )

    # With the string_to_boolean argparse type arg,
    # you can supply the flag for True, omit it for False, or pass True/False explicitly
    parser.add_argument(
        "--dry-run",
        type=string_to_boolean,
        nargs="?",
        const=True,
        default=False,
        help=(
            "If set to True, do not submit tasks to Azure Batch. Only print what would be done."
            "Pass --dry-run True or --dry-run False to set explicitly; omit to use default (False)."
        ),
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
