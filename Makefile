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

dagster_push_prod: dagster_push
	uv run https://raw.githubusercontent.com/CDCgov/cfa-dagster/refs/heads/main/scripts/update_code_location.py \
    	--registry_image cfaprdbatchcr.azurecr.io/pyrenew-hew:dagster_latest

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

post_process: config
	uv run python pipelines/postprocess_forecast_batches.py \
    	--input "./blobfuse/mounts/pyrenew-hew-prod-output/${FORECAST_DATE}_forecasts" \
    	--output "./blobfuse/mounts/nssp-etl/gold/${FORECAST_DATE}_forecasts.parquet" \
		${ARGS}
