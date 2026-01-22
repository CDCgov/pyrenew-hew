"""
Set up a multi-location, multi-disease run of pyrenew-hew
with the PyRenew model family on Azure Batch.
"""

import argparse
import itertools
from pathlib import Path

from rich.console import Console

from pipelines.batch.common_batch_utils import (
    DEFAULT_EXCLUDED_LOCATIONS,
    create_job_details_table,
    get_filtered_locations,
    get_pyrenew_container_settings,
    print_job_header,
    string_to_boolean,
    submit_batch_job,
    validate_diseases,
)
from pyrenew_hew.utils import validate_hew_letters


def main(
    model_letters: str,
    job_id: str,
    pool_id: str,
    diseases: str | list[str],
    output_subdir: str | Path = "./",
    additional_forecast_letters: str = "",
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
    Set up and submit a PyRenew model job to Azure Batch.

    Parameters
    ----------
    model_letters
        Model letters for pyrenew model (e.g. 'he', 'e', 'hew').
    job_id
        Name for the Batch job.
    pool_id
        Azure Batch pool on which to run the job.
    diseases
        Name(s) of disease(s) to run as part of the job,
        as a single string (one disease) or a list of strings.
        Supported values are 'COVID-19', 'Influenza', and 'RSV'.
    output_subdir
        Subdirectory of the output blob storage container
        in which to save results.
    additional_forecast_letters
        Forecast the following signals even if they were not fit.
        Fit signals are always forecast. If empty, uses model_letters.
    container_image_name
        Name of the container to use for the job.
        Default 'pyrenew-hew'.
    container_image_version
        Version of the container to use. Default 'latest'.
    n_training_days
        Number of training days of data to use for model fitting.
        Default 150.
    exclude_last_n_days
        Number of days of available data to exclude from fitting.
        Default 1.
    rng_key
        Random number generator seed for reproducibility.
        Default 12345.
    locations_include
        List of two-letter USPS location abbreviations for locations
        to include in the job. If None, use all available locations.
    locations_exclude
        List of additional two letter USPS location abbreviations to
        exclude from the job. If None, only default locations excluded.
    test
        Is this a testing run? Default False.
    dry_run
        If set, do not submit tasks to Azure Batch. Default False.

    Returns
    -------
    None
    """
    # Validate inputs
    disease_list = diseases if isinstance(diseases, list) else [diseases]
    validate_diseases(disease_list)
    validate_hew_letters(model_letters)
    additional_forecast_letters = additional_forecast_letters or model_letters
    validate_hew_letters(additional_forecast_letters)

    # Sampling parameters
    n_warmup = 200 if test else 1000
    n_samples = 200 if test else 500
    n_chains = 2 if test else 4

    # Get filtered locations
    all_locations = get_filtered_locations(locations_include, locations_exclude)
    all_exclusions = list(set(DEFAULT_EXCLUDED_LOCATIONS + (locations_exclude or [])))

    # Container setup
    container_image = f"ghcr.io/cdcgov/{container_image_name}:{container_image_version}"
    container_settings = get_pyrenew_container_settings(container_image, test)

    # Build command
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

    # Print job details
    print_job_header("pyrenew")
    console = Console()

    details = {
        "job_id": job_id,
        "pool_id": pool_id,
        "model_family": "pyrenew",
        "model_letters": model_letters,
        "additional_forecast_letters": additional_forecast_letters,
        "diseases": ", ".join(disease_list),
        "output_subdir": str(output_subdir),
        "container_image": container_image,
        "training_days": n_training_days,
        "exclude_last_n_days": exclude_last_n_days,
        "rng_key": rng_key,
        "locations_included": all_locations,
        "excluded_locations": ", ".join(all_exclusions),
        "test": test,
        "dry_run": dry_run,
    }

    table = create_job_details_table(details)
    console.print(table)

    # Create tasks
    tasks = []
    for disease, loc in itertools.product(disease_list, all_locations):
        task_id = f"{job_id}-{loc}-{disease}-prod"
        command = base_call.format(
            loc=loc,
            disease=disease,
            output_dir=str(Path("output", output_subdir)),
        )
        tasks.append((task_id, command, container_settings))

    # Submit job
    submit_batch_job(job_id, pool_id, tasks, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up PyRenew model job on Azure Batch"
    )
    parser.add_argument(
        "--model-letters",
        type=str,
        required=True,
        help="Model letters for pyrenew model (e.g. 'he', 'e', 'hew').",
    )
    parser.add_argument(
        "--job-id", type=str, required=True, help="Name for the Azure batch job"
    )
    parser.add_argument(
        "--pool-id",
        type=str,
        help="Name of the Azure batch pool on which to run the job",
        default="pyrenew-pool",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        default="COVID-19 Influenza RSV",
        help=(
            "Name(s) of disease(s) to run as part of the job, "
            "as a whitespace-separated string. Default 'COVID-19 Influenza RSV'."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        help="Subdirectory of the output blob storage container in which to save results.",
        default="./",
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
        help="Number of 'training days' of observed data to use for model fitting.",
        default=150,
    )
    parser.add_argument(
        "--exclude-last-n-days",
        type=int,
        help="Number of days to drop from the end of the timeseries of observed data.",
        default=1,
    )
    parser.add_argument(
        "--rng-key",
        type=int,
        help="Random number generator seed for reproducibility (default: 12345).",
        default=12345,
    )
    parser.add_argument(
        "--locations-include",
        type=str,
        help=(
            "Two-letter USPS location abbreviations to include in the job, "
            "as a whitespace-separated string."
        ),
        default=None,
    )
    parser.add_argument(
        "--locations-exclude",
        type=str,
        help=(
            f"Additional two-letter USPS location abbreviations to exclude from the job, "
            f"as a whitespace-separated string. These will be combined with the default "
            f"exclusions ({' '.join(DEFAULT_EXCLUDED_LOCATIONS)})."
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
        "--test",
        type=string_to_boolean,
        nargs="?",
        const=True,
        default=False,
        help="Run in test mode (default: False).",
    )
    parser.add_argument(
        "--dry-run",
        type=string_to_boolean,
        nargs="?",
        const=True,
        default=False,
        help="If set to True, do not submit tasks to Azure Batch. Only print what would be done.",
    )

    args = parser.parse_args()
    args.diseases = args.diseases.split()
    args.locations_include = (
        args.locations_include.split() if args.locations_include else None
    )
    args.locations_exclude = (
        args.locations_exclude.split() if args.locations_exclude else None
    )
    if args.additional_forecast_letters is None:
        args.additional_forecast_letters = args.model_letters

    main(**vars(args))
