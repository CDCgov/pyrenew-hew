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

# ----------- #
# Help Target #
# ----------- #

help:
	@echo "Usage: make [target] [ARGS]"
	@echo ""
	@echo "Container Build Targets: "
	@echo "  container_build     : Build the container image"
	@echo "  container_tag       : Tag the container image"
	@echo "  ghcr_login          : Log in to the Github Container Registry. Requires GH_USERNAME and GH_PAT env vars"
	@echo "  container_push      : Push the container image to the Azure Container Registry"
	@echo ""
	@echo "Model Fit Targets: "
	@echo "  run_timeseries      : Run the timeseries model fit job"
	@echo "  run_e_model         : Run an e model fit job"
	@echo "  run_h_models        : Run an h model fit job"
	@echo "  run_he_model        : Run an he model fit job"
	@echo "  run_hw_model        : Run an hw model fit job"
	@echo "  run_hew_model       : Run an hew model fit job"
	@echo "  post_process        : Post-process the forecast batches"
	@echo ""
	@echo "Toggle default forecasting parameters with the following syntax:"
	@echo "  make <target> TEST=True DRY_RUN=True MODEL_LETTERS=<letters> FORECAST_DATE=<date>"
	@echo ""
	@echo "For example, to run the timeseries model in production, you can simply type:"
	@echo "  make run_timeseries"
	@echo ""
	@echo "To run the pyrenew-e model in test mode with a dry run for a custom date:"
	@echo "  make run_e_model TEST=True DRY_RUN=True FORECAST_DATE=2025-07-01"
	@echo ""
	@echo "To run the pyrenew-hew model and output to pyrenew-test-output:"
	@echo "  make run_hew_model TEST=True MODEL_LETTERS=hew"
	@echo ""
	@echo "Any additional flags can be passed with ARGS, for example:"
	@echo "  make run_hew_model ARGS=\"--locations-include 'NY GA'\""
	@echo ""
	@echo "Passing a flag through ARGS will also override the flags set previously."

# ----------------------- #
# Container Build Targets
# ----------------------- #

container_build: ghcr_login
	$(ENGINE) build . -t $(CONTAINER_NAME) -f $(CONTAINERFILE)

container_tag:
	$(ENGINE) tag $(CONTAINER_NAME) $(CONTAINER_REMOTE_NAME)

ghcr_login:
	echo $(GH_PAT) | $(ENGINE) login ghcr.io -u $(GH_USERNAME) --password-stdin

container_push: container_tag ghcr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)

# ---------------- #
# Model Fit Targets
# ---------------- #

run_timeseries:
	uv run python pipelines/batch/setup_job.py \
		--model-family timeseries \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "e" \
		--job-id "pyrenew-e-prod_${FORECAST_DATE}_t" \
		--pool-id pyrenew-pool \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		$(ARGS)

run_e_model:
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "e" \
		--job-id "pyrenew-e-prod_${FORECAST_DATE}" \
		--pool-id pyrenew-pool \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		$(ARGS)

run_h_model:
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "h" \
		--job-id "pyrenew-h-prod_${FORECAST_DATE}" \
		--pool-id pyrenew-pool \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		$(ARGS)

run_he_model:
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "he" \
		--job-id "pyrenew-he-prod_${FORECAST_DATE}" \
		--pool-id pyrenew-pool \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		$(ARGS)

run_hw_model:
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "hw" \
		--job-id "pyrenew-hw-prod_${FORECAST_DATE}" \
		--pool-id pyrenew-pool-32gb \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		$(ARGS)

run_hew_model:
	uv run python pipelines/batch/setup_job.py \
		--model-family pyrenew \
		--output-subdir "${FORECAST_DATE}_forecasts" \
		--model-letters "hew" \
		--job-id "pyrenew-hew-prod_${FORECAST_DATE}" \
		--pool-id pyrenew-pool-32gb \
		--test "$(TEST)" \
		--dry-run "$(DRY_RUN)" \
		$(ARGS)

post_process:
	uv run python pipelines/postprocess_forecast_batches.py \
    	--input "./blobfuse/mounts/pyrenew-hew-prod-output/${FORECAST_DATE}_forecasts" \
    	--output "./blobfuse/mounts/nssp-etl/gold/${FORECAST_DATE}_forecasts.parquet" \
		${ARGS}
