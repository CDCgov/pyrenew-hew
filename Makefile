.PHONY: help container_build container_tag ghcr_login container_push run_timeseries run_e_model run_h_models post_process

# Build parameters

ifndef ENGINE
ENGINE = docker
endif

ifndef CONTAINER_NAME
CONTAINER_NAME = pyrenew-hew
endif

ifndef CONTAINERFILE
CONTAINERFILE = Containerfile
endif

ifndef CONTAINER_REMOTE_NAME
CONTAINER_REMOTE_NAME = ghcr.io/cdcgov/$(CONTAINER_NAME):latest
endif

# Forecasting paramters

ifndef FORECAST_DATE
FORECAST_DATE = $(shell date +%Y-%m-%d)
endif

ifndef FORECAST_YEAR_MONTH
FORECAST_YEAR_MONTH = $(shell date +%y%m)
endif


help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  container_build     : Build the container image"
	@echo "  container_tag       : Tag the container image"
	@echo "  ghcr_login          : Log in to the Github Container Registry. Requires GH_USERNAME and GH_PAT env vars"
	@echo "  container_push      : Push the container image to the Azure Container Registry"
	@echo "  run_timeseries      : Run the timeseries forecasting job"
	@echo "  run_e_model         : Run the e_model forecasting job"
	@echo "  run_h_models        : Run the h_models forecasting job"
	@echo "  post_process        : Post-process the forecast batches"
	@echo "  help                : Show this help message"

# Build

container_build: ghcr_login
	$(ENGINE) build . -t $(CONTAINER_NAME) -f $(CONTAINERFILE)

container_tag:
	$(ENGINE) tag $(CONTAINER_NAME) $(CONTAINER_REMOTE_NAME)

ghcr_login:
	echo $(GH_PAT) | $(ENGINE) login ghcr.io -u $(GH_USERNAME) --password-stdin

container_push: container_tag ghcr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)

# Forecasting

run_timeseries:
	uv run python pipelines/batch/setup_job.py \
		--model-family timeseries \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model_letters "e" \
		--job_id "pyrenew-e-prod${FORECAST_YEAR_MONTH}t" \
		--pool_id pyrenew-pool

run_e_model:
	uv run python pipelines/batch/setup_job.py \
		--model-family e_model \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model_letters "e" \
		--job_id "pyrenew-e-prod${FORECAST_YEAR_MONTH}e" \
		--pool_id pyrenew-pool

run_h_models:
	uv run python pipelines/batch/setup_job.py \
		--model-family h_models \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model_letters "h" \
		--job_id "pyrenew-h-prod${FORECAST_YEAR_MONTH}h" \
		--pool_id pyrenew-pool

post_process:	
	uv run python pipelines/postprocess_forecast_batches.py \
    	--input "./blobfuse/mounts/pyrenew-hew-prod-output/${FORECAST_DATE}_forecasts" \
    	--output "./blobfuse/mounts/nssp-etl/gold/${FORECAST_DATE}.parquet"