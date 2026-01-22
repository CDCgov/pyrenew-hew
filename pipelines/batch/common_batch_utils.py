"""
Common utilities for Azure Batch job submission across different model types.
"""

from pathlib import Path
from typing import Any

from azure.batch import models
from azuretools.auth import EnvCredentialHandler
from azuretools.client import get_batch_service_client
from azuretools.job import create_job
from azuretools.task import get_container_settings, get_task_config
from forecasttools import location_table
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Locations that are always excluded due to lack of NSSP ED visit data
DEFAULT_EXCLUDED_LOCATIONS = ["AS", "GU", "MP", "PR", "UM", "VI"]

# Supported diseases across all model families
SUPPORTED_DISEASES = ["COVID-19", "Influenza", "RSV"]


def validate_diseases(disease_list: list[str]) -> None:
    """
    Validate that all diseases in the list are supported.

    Parameters
    ----------
    disease_list
        List of disease names to validate.

    Raises
    ------
    ValueError
        If any disease is not in the supported list.
    """
    invalid_diseases = set(disease_list) - set(SUPPORTED_DISEASES)
    if invalid_diseases:
        raise ValueError(
            f"Unsupported diseases: {', '.join(invalid_diseases)}; "
            f"supported diseases are: {', '.join(SUPPORTED_DISEASES)}"
        )


def get_filtered_locations(
    locations_include: list[str] | None = None,
    locations_exclude: list[str] | None = None,
) -> list[str]:
    """
    Get list of locations after applying inclusion and exclusion filters.

    Parameters
    ----------
    locations_include
        List of two-letter USPS location abbreviations to include.
        If None, use all available locations.
    locations_exclude
        List of additional two-letter USPS location abbreviations to exclude.
        These will be combined with the default exclusions.

    Returns
    -------
    list[str]
        Filtered list of location abbreviations.
    """
    loc_abbrs = location_table.get_column("short_name").to_list()
    locations_include = locations_include or loc_abbrs
    locations_exclude = locations_exclude or []

    # Combine default exclusions with any additional exclusions
    all_exclusions = list(set(DEFAULT_EXCLUDED_LOCATIONS + locations_exclude))

    all_locations = [
        loc
        for loc in loc_abbrs
        if loc not in all_exclusions and loc in locations_include
    ]

    return all_locations


def get_pyrenew_container_settings(
    container_image: str,
    test: bool = False,
) -> models.TaskContainerSettings:
    """
    Get container settings for pyrenew-hew jobs.

    Parameters
    ----------
    container_image
        Full container image name including registry and version.
    test
        Whether this is a test run (determines output container).

    Returns
    -------
    models.TaskContainerSettings
        Container settings for the task.
    """
    pyrenew_hew_output_container = (
        "pyrenew-test-output" if test else "pyrenew-hew-prod-output"
    )

    return get_container_settings(
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


def print_job_header(model_family: str) -> None:
    """
    Print a formatted job header for the console.

    Parameters
    ----------
    model_family
        Name of the model family (e.g., 'pyrenew', 'epiautogp', 'timeseries').
    """
    console = Console()
    console.print(
        Panel.fit(
            f"[bold magenta]🚀 pyrenew-hew: Azure Batch Job Submission ({model_family}) 🚀",
            border_style="magenta",
        )
    )


def create_job_details_table(details: dict[str, Any]) -> Table:
    """
    Create a Rich Table with job details.

    Parameters
    ----------
    details
        Dictionary of job detail key-value pairs to display.

    Returns
    -------
    Table
        Rich Table object with job details.
    """
    table = Table(show_header=False, pad_edge=False)

    for key, value in details.items():
        if key == "locations_included":
            # Format locations 5 per line
            loc_lines = [", ".join(value[i : i + 5]) for i in range(0, len(value), 5)]
            table.add_row("Locations Included", loc_lines[0] if loc_lines else "")
            for loc_line in loc_lines[1:]:
                table.add_row("", loc_line)
        elif key == "test" or key == "dry_run":
            # Style boolean values
            styled_value = (
                "[bold green]True[/bold green]" if value else "[grey50]False[/grey50]"
            )
            table.add_row(key.replace("_", " ").title(), styled_value)
        else:
            # Standard key-value display
            display_key = key.replace("_", " ").title()
            display_value = ", ".join(value) if isinstance(value, list) else str(value)
            table.add_row(display_key, display_value)

    return table


def submit_batch_job(
    job_id: str,
    pool_id: str,
    tasks: list[tuple[str, str, models.TaskContainerSettings]],
    dry_run: bool = False,
) -> None:
    """
    Submit a job and tasks to Azure Batch.

    Parameters
    ----------
    job_id
        Unique identifier for the batch job.
    pool_id
        Azure Batch pool on which to run the job.
    tasks
        List of tuples containing (task_id, command, container_settings).
    dry_run
        If True, only print what would be done without submitting.

    Returns
    -------
    None
    """
    console = Console()

    if dry_run:
        console.print("Dry run mode enabled. No tasks will be submitted.")
        console.print(f"Would create job: {job_id}")
        console.print(f"Would submit {len(tasks)} tasks to pool: {pool_id}")
        console.print("Closing...")
        return None

    print("")
    print("Using environment credentials to authenticate with Azure Batch...")
    creds = EnvCredentialHandler()
    client = get_batch_service_client(creds)

    print("Submitting job to Azure Batch...")
    job = models.JobAddParameter(
        id=job_id,
        pool_info=models.PoolInformation(pool_id=pool_id),
    )
    create_job(client, job)

    print(f"Submitting {len(tasks)} tasks...")
    for task_id, base_call, container_settings in tasks:
        task = get_task_config(
            task_id,
            base_call=base_call,
            container_settings=container_settings,
            log_blob_container="pyrenew-hew-logs",
            log_blob_account=creds.azure_blob_storage_account,
            log_subdir=job_id,
            log_compute_node_identity_reference=creds.compute_node_identity_reference,
        )
        client.task.add(job_id, task)

    print(f"Successfully submitted job {job_id} with {len(tasks)} tasks.")


def string_to_boolean(value: str | bool) -> bool:
    """
    Convert string to boolean for argparse type handling.

    Parameters
    ----------
    value
        String or boolean value to convert.

    Returns
    -------
    bool
        Boolean representation of the input.

    Raises
    ------
    ValueError
        If the string value cannot be interpreted as a boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")
