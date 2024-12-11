.PHONY: help container_build container_tag acr_login container_push dep_container_build dep_container_tag dep_container_push

ifndef ENGINE
ENGINE = docker
endif

ifndef DEP_CONTAINER_NAME
DEP_CONTAINER_NAME = pyrenew-hew-dependencies
endif

ifndef DEP_CONTAINERFILE
DEP_CONTAINERFILE = Containerfile.dependencies
endif

ifndef DEP_CONTAINER_REMOTE_NAME
DEP_CONTAINER_REMOTE_NAME = $(ACR_TAG_PREFIX)$(DEP_CONTAINER_NAME):latest
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
	@echo "  dep_container_build : Build the dependencies container image"
	@echo "  dep_container_tag   : Tag the dependencies container image"
	@echo "  dep_container_push  : Push the dependencies container image to the Azure Container Registry"

dep_container_build:
	$(ENGINE) build . -t $(DEP_CONTAINER_NAME) -f $(DEP_CONTAINERFILE)

dep_container_tag:
	$(ENGINE) tag $(DEP_CONTAINER_NAME) $(DEP_CONTAINER_REMOTE_NAME)

container_build: acr_login
	$(ENGINE) build . -t $(CONTAINER_NAME) -f $(CONTAINERFILE)

container_tag:
	$(ENGINE) tag $(CONTAINER_NAME) $(CONTAINER_REMOTE_NAME)

acr_login:
	az acr login -n $(AZURE_CONTAINER_REGISTRY_ACCOUNT)

dep_container_push: dep_container_tag acr_login
	$(ENGINE) push $(DEP_CONTAINER_REMOTE_NAME)

container_push: container_tag acr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)
