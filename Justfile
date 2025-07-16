# Variables
ENGINE := "docker"
CONTAINER_NAME := "pyrenew-hew"
CONTAINERFILE := "Containerfile"
CONTAINER_REMOTE_NAME := "ghcr.io/cdcgov/{{CONTAINER_NAME}}:latest"

FORECAST_DATE := `date +%Y-%m-%d`
FORECAST_YEAR_MONTH := `date +%y%m`

# Recipes
help:
    @echo "Usage: just <recipe>"
    @echo ""
    @echo "Available recipes:"
    @echo "-- Container Build Recipes --"
    @echo "  container_build     Build the container image"
    @echo "  container_tag       Tag the container image"
    @echo "  ghcr_login          Log in to the Github Container Registry (requires GH_USERNAME and GH_PAT env vars)"
    @echo "  container_push      Push the container image to the Github Container Registry"
    @echo ""
    @echo "-- Model Inference Recipes --"
    @echo "  run_timeseries      Run the timeseries forecasting job"
    @echo "  run_e_model         Run the e_model forecasting job"
    @echo "  run_h_models        Run the h_models forecasting job"
    @echo "  post_process        Post-process the forecast batches"
    @echo "  help                Show this help message"

# Container management recipes
container_build: ghcr_login
    {{ENGINE}} build . -t {{CONTAINER_NAME}} -f {{CONTAINERFILE}}

container_tag:
    {{ENGINE}} tag {{CONTAINER_NAME}} {{CONTAINER_REMOTE_NAME}}

ghcr_login:
    echo $GH_PAT | {{ENGINE}} login ghcr.io -u $GH_USERNAME --password-stdin

container_push: container_tag ghcr_login
    {{ENGINE}} push {{CONTAINER_REMOTE_NAME}}

# Model inference run receipes
run_timeseries test?=false:
    uv run python pipelines/batch/setup_job.py \
        --model-family timeseries \
        --output-subdir "{{FORECAST_DATE}}_jon_forecasts" \
        --model_letters "e" \
        --job_id "pyrenew-e-prod_{{FORECAST_YEAR_MONTH}}_jon_t" \
        --pool_id pyrenew-pool \
        --test {{test}}

run_e_model test?=false:
    uv run python pipelines/batch/setup_job.py \
        --model-family e_model \
        --output-subdir "{{FORECAST_DATE}}_jon_forecasts" \
        --model_letters "e" \
        --job_id "pyrenew-e-prod_{{FORECAST_YEAR_MONTH}}_jon_e" \
        --pool_id pyrenew-pool \
        --test {{test}}

run_h_models test?=false:
    uv run python pipelines/batch/setup_job.py \
        --model-family h_models \
        --output-subdir "{{FORECAST_DATE}}_jon_forecasts" \
        --model_letters "h" \
        --job_id "pyrenew-h-prod{{FORECAST_YEAR_MONTH}}_jon_h" \
        --pool_id pyrenew-pool \
        --test {{test}}

post_process:
    uv run python pipelines/postprocess_forecast_batches.py \
        --input "./blobfuse/mounts/pyrenew-hew-prod-output/{{FORECAST_DATE}}_jon_forecasts" \
        --output "./blobfuse/mounts/nssp-etl/gold/{{FORECAST_DATE}}_jon.parquet"
