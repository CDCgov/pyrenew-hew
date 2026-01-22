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
	@echo "  dagster_push        : Push the dagster container image to the Azure Container Registry and code location"
	@echo "  container_tag       : Tag the container image"
	@echo "  ghcr_login          : Log in to the Github Container Registry. Requires GH_USERNAME and GH_PAT env vars"
	@echo "  container_push      : Push the container image to the Azure Container Registry"
	@echo ""
	@echo "Model Fit Targets: "
	@echo "  acc                 : Run the Azure Command Center for routine production jobs"
	@echo "  run_timeseries      : Run the timeseries model fit job"
	@echo "  run_e_model         : Run an e model fit job"
	@echo "  run_h_models        : Run an h model fit job"
	@echo "  run_he_model        : Run an he model fit job"
	@echo "  run_hw_model        : Run an hw model fit job"
	@echo "  run_hew_model       : Run an hew model fit job"
	@echo "  post_process        : Post-process the forecast batches"
	@echo ""
	@echo "Toggle default forecasting parameters with the following syntax:"
	@echo "  make <target> TEST=True DRY_RUN=True RNG_KEY=54321 MODEL_LETTERS=<letters> FORECAST_DATE=<date>"
	@echo ""
	@echo "For example, to run the timeseries model in production, you can simply type:"
	@echo "  make run_timeseries"
	@echo ""
	@echo "To run the pyrenew-e model in test mode with a dry run for a custom date:"
	@echo "  make run_e_model TEST=True DRY_RUN=True FORECAST_DATE=2025-07-01"
	@echo ""
	@echo "To run the pyrenew-hew model and output to pyrenew-test-output:"
	@echo "  make run_hew_model TEST=True MODEL_LETTERS=hew"
	@echo "To use a custom container registry, image, and tag:"
	@echo "  make run_hew_model CONTAINER_REGISTRY=<custom_registry> CONTAINER_IMAGE_NAME=<custom_image> CONTAINER_IMAGE_VERSION=<custom_tag>"
	@echo ""
	@echo "Any additional flags can be passed with ARGS, for example:"
	@echo "  make run_hew_model ARGS=\"--locations-include 'NY GA'\""
	@echo ""
	@echo "Passing a flag through ARGS will also override the flags set previously."

#------------------------ #
# Blobfuse Mount Targets
# ----------------------- #

mount:
	sudo bash -c "source ./blobfuse/mount.sh"

unmount:
	sudo bash -c "source ./blobfuse/cleanup.sh"

# ----------------------- #
# Container Build Targets
# ----------------------- #

container_build: ghcr_login
	$(ENGINE) build . -t $(CONTAINER_IMAGE_NAME) -f $(CONTAINERFILE)

dagster_build:
	docker build -t cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest -f Containerfile .

dagster:
	uv run dagster_defs.py

dagster_push: dagster_build
	az login --identity && \
	az acr login -n cfaprdbatchcr && \
	docker push "cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest" 

container_tag:
	$(ENGINE) tag $(CONTAINER_IMAGE_NAME) $(CONTAINER_REMOTE_NAME)

ghcr_login:
	echo $(GH_PAT) | $(ENGINE) login ghcr.io -u $(GH_USERNAME) --password-stdin

container_push: CONTAINER_IMAGE_VERSION ghcr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)

config:
	bash -c "source ./azureconfig.sh"

# ---------------- #
# Model Fit Targets
# ---------------- #

acc: mount config
	uv run pipelines/azure_command_center.py

# Auto-set TEST/ENVIRONMENT based on each other
ifeq ($(shell echo $(ENVIRONMENT) | tr A-Z a-z),test)
override TEST = True
endif
ifeq ($(shell echo $(TEST) | tr A-Z a-z),true)
override ENVIRONMENT = test
endif

run_timeseries: config
	uv run python pipelines/batch/setup_job.py \
		--model-family timeseries \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "e" \
		--locations-exclude "WY" \
		--job-id "timeseries-e-${ENVIRONMENT}_${FORECAST_DATE}_makefile" \
		--pool-id pyrenew-pool \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		--container-registry "$(CONTAINER_REGISTRY)" \
		--container-image-name "$(CONTAINER_IMAGE_NAME)" \
		--container-image-version "$(CONTAINER_IMAGE_VERSION)" \
		$(ARGS)

run_e_model: config
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "e" \
		--locations-exclude "WY" \
		--job-id "pyrenew-e-${ENVIRONMENT}_${FORECAST_DATE}_makefile" \
		--pool-id pyrenew-pool \
		--rng-key "$(RNG_KEY)" \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		--container-registry "$(CONTAINER_REGISTRY)" \
		--container-image-name "$(CONTAINER_IMAGE_NAME)" \
		--container-image-version "$(CONTAINER_IMAGE_VERSION)" \
		$(ARGS)

run_h_model: config
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "h" \
		--job-id "pyrenew-h-${ENVIRONMENT}_${FORECAST_DATE}_makefile" \
		--pool-id pyrenew-pool \
		--rng-key "$(RNG_KEY)" \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		--container-registry "$(CONTAINER_REGISTRY)" \
		--container-image-name "$(CONTAINER_IMAGE_NAME)" \
		--container-image-version "$(CONTAINER_IMAGE_VERSION)" \
		$(ARGS)

run_he_model: config
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "he" \
		--locations-exclude "WY" \
		--job-id "pyrenew-he-${ENVIRONMENT}_${FORECAST_DATE}_makefile" \
		--pool-id pyrenew-pool \
		--rng-key "$(RNG_KEY)" \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		--container-registry "$(CONTAINER_REGISTRY)" \
		--container-image-name "$(CONTAINER_IMAGE_NAME)" \
		--container-image-version "$(CONTAINER_IMAGE_VERSION)" \
		$(ARGS)

run_hw_model: config
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "hw" \
		--diseases "COVID-19" \
		--job-id "pyrenew-hw-${ENVIRONMENT}_${FORECAST_DATE}_makefile" \
		--pool-id pyrenew-pool \
		--rng-key "$(RNG_KEY)" \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		--container-registry "$(CONTAINER_REGISTRY)" \
		--container-image-name "$(CONTAINER_IMAGE_NAME)" \
		--container-image-version "$(CONTAINER_IMAGE_VERSION)" \
		$(ARGS)

run_hew_model: config
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "hew" \
		--locations-exclude "WY" \
		--diseases "COVID-19" \
		--job-id "pyrenew-hew-${ENVIRONMENT}_${FORECAST_DATE}_makefile" \
		--pool-id pyrenew-pool \
		--rng-key "$(RNG_KEY)" \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		--container-registry "$(CONTAINER_REGISTRY)" \
		--container-image-name "$(CONTAINER_IMAGE_NAME)" \
		--container-image-version "$(CONTAINER_IMAGE_VERSION)" \
		$(ARGS)

post_process: config
	uv run python pipelines/postprocess_forecast_batches.py \
    	--input "./blobfuse/mounts/pyrenew-hew-prod-output/${FORECAST_DATE}_forecasts" \
    	--output "./blobfuse/mounts/nssp-etl/gold/${FORECAST_DATE}_forecasts.parquet" \
		${ARGS}
