#!/usr/bin/env nextflow

params.date_string = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd")
params.month_date = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM")
params.h_letters = "hew"
params.h_pool = "pyrenew-pool-32gb"

process config {
    """
    sudo apt-get update
    chmod +x ./mount.sh
    ./mount.sh
    """
}

process timeseries {
    """
    uv run python pipelines/batch/setup_job.py \
        --model-family timeseries \
        --output-subdir "${params.date_string}_forecasts" \
        --model-letter "e" \
        --output "pyrenew-e-prod${params.month_date}t" \
        --pool pyrenew-pool
    """
}

process run_e_model {
    """
    uv run python pipelines/batch/setup_job.py \
        --model-family pyrenew \
        --output-subdir "${params.date_string}_forecasts" \
        --model-letter "e" \
        --output "pyrenew-e-prod${params.month_date}" \
        --pool pyrenew-pool
    """
}

process run_h_models {
    """
    uv run python pipelines/batch/setup_job.py \
        --model-family pyrenew \
        --output-subdir "${params.date_string}_forecasts" \
        --model-letter "${params.h_letters}" \
        --output "pyrenew-h-prod${params.month_date}" \
        --pool ${params.h_pool}
    """
}

process post_process {
    """
    uv run python pipelines/postprocess_forecast_batches.py \
    --input "./blobfuse/mounts/pyrenew-hew-prod-output/2025-${params.month_day}_forecasts" \
    --output "./blobfuse/mounts/nssp-etl/gold/${params.date_string}.parquet"
    """
}

workflow {
    config()
    timeseries()
    run_e_model()
    run_h_models()
    post_process()
}