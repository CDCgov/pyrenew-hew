.PHONY: help container_build container_tag ghcr_login container_push

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


help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  container_build     : Build the container image"
	@echo "  container_tag       : Tag the container image"
	@echo "  ghcr_login          : Log in to the Github Container Registry. Requires GH_USERNAME and GH_PAT env vars"
	@echo "  container_push      : Push the container image to the Azure Container Registry"

container_build: ghcr_login
	$(ENGINE) build . -t $(CONTAINER_NAME) -f $(CONTAINERFILE)

container_tag:
	$(ENGINE) tag $(CONTAINER_NAME) $(CONTAINER_REMOTE_NAME)

ghcr_login:
	echo $(GH_PAT) | $(ENGINE) login ghcr.io -u $(GH_USERNAME) --password-stdin

container_push: container_tag ghcr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)
