#!/usr/bin/env nextflow

params.date_string = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd")
params.year_month = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM")
params.h_letters = "hew"
params.h_pool = "pyrenew-pool-32gb"

// process blobfuse {
//     """
//     sudo apt-get update
//     git clone https://github.com/cdcent/cfa-blobfuse-tutorial.git
//     ./mount.sh
//     """
// }

process timeseries {
    """
    uv run python pipelines/batch/setup_job.py \
        --model-family timeseries \
        --output-subdir "${params.date_string}_forecasts" \
        --model-letters "e" \
        --job_id "pyrenew-e-prod${params.year_month}t" \
        --pool_id pyrenew-pool
    """
}

process run_e_model {
    """
    uv run python pipelines/batch/setup_job.py \
        --model-family pyrenew \
        --output-subdir "${params.date_string}_forecasts" \
        --model-letters "e" \
        --job_id "pyrenew-e-prod${params.year_month}" \
        --pool_id pyrenew-pool
    """
}

process run_h_models {
    """
    uv run python pipelines/batch/setup_job.py \
        --model-family pyrenew \
        --output-subdir "${params.date_string}_forecasts" \
        --model-letters "${params.h_letters}" \
        --job_id "pyrenew-h-prod${params.year_month}" \
        --pool_id ${params.h_pool}
    """
}

process post_process {
    """
    uv run python pipelines/postprocess_forecast_batches.py \
    --input "./blobfuse/mounts/pyrenew-hew-prod-output/${params.date_string}_forecasts" \
    --output "./blobfuse/mounts/nssp-etl/gold/${params.date_string}.parquet"
    """
}

workflow {
    // blobfuse()
    timeseries()
    run_e_model()
    run_h_models()
    post_process()
}