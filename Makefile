.PHONY: help container_build container_tag ghcr_login container_push run_timeseries run_e_model run_h_models post_process run_he_model run_hw_model acc mount unmount config dagster dagster_build

# Build parameters

ifndef ENGINE
ENGINE = docker
endif

ifndef CONTAINER_REGISTRY
CONTAINER_REGISTRY = ghcr.io/cdcgov
endif

ifndef CONTAINER_IMAGE_NAME
CONTAINER_IMAGE_NAME = pyrenew-hew
endif

ifndef CONTAINER_IMAGE_VERSION
CONTAINER_IMAGE_VERSION = latest
endif

ifndef CONTAINER_REMOTE_NAME
CONTAINER_REMOTE_NAME = $(CONTAINER_REGISTRY)/$(CONTAINER_IMAGE_NAME):$(CONTAINER_IMAGE_VERSION)
endif

ifndef CONTAINERFILE
CONTAINERFILE = Containerfile
endif

# Model Fit Parameters

ifndef FORECAST_DATE
FORECAST_DATE = $(shell date +%Y-%m-%d)
endif

ifndef TEST
TEST = False
endif

ifndef DRY_RUN
DRY_RUN = False
endif

ifndef ENVIRONMENT
ENVIRONMENT = prod
endif

ifndef RNG_KEY
RNG_KEY = 12345
endif

# ----------- #
# Help Target #
# ----------- #

help:
	@echo "Usage: make [target] [ARGS]"
	@echo ""

	@echo "Blobfuse Mount Targets: "
	@echo "  mount              : Mount blob storage containers using blobfuse2"
	@echo "  unmount            : Unmount blob storage containers and clean up"
	@echo ""
	@echo "Container Build Targets: "
	@echo "  container_build     : Build the container image"
	@echo "  dagster             : Run dagster definitions locally"
	@echo "  dagster_build       : Build the dagster container image"
	@echo "  dagster_push        : Push the dagster container image to the Azure Container Registry"
	@echo "  dagster_push_prod   : Push the dagster container image to the Azure Container Registry and code location for production"
	@echo "  container_tag       : Tag the container image"
	@echo "  ghcr_login          : Log in to the Github Container Registry. Requires GH_USERNAME and GH_PAT env vars"
	@echo "  container_push      : Push the container image to the Azure Container Registry"
	@echo ""
	@echo "Model Fit Targets: "
	@echo "  acc                 : Run the Azure Command Center for routine production jobs"

post_process: config
	uv run python pipelines/postprocess_forecast_batches.py \
    	--input "./blobfuse/mounts/pyrenew-hew-prod-output/${FORECAST_DATE}_forecasts" \
    	--output "./blobfuse/mounts/nssp-etl/gold/${FORECAST_DATE}_forecasts.parquet" \
		${ARGS}
