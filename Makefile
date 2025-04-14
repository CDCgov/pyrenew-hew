.PHONY: help container_build container_tag acr_login container_push

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
CONTAINER_REMOTE_NAME = $(ACR_TAG_PREFIX)$(CONTAINER_NAME):latest
endif


help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  container_build     : Build the container image"
	@echo "  container_tag       : Tag the container image"
	@echo "  acr_login           : Log in to the Azure Container Registry"
	@echo "  container_push      : Push the container image to the Azure Container Registry"

container_build: acr_login
	$(ENGINE) build . -t $(CONTAINER_NAME) -f $(CONTAINERFILE)

container_tag:
	$(ENGINE) tag $(CONTAINER_NAME) $(CONTAINER_REMOTE_NAME)

acr_login:
	az acr login -n $(AZURE_CONTAINER_REGISTRY_ACCOUNT)

container_push: container_tag acr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)
