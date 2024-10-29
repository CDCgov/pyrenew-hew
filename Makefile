.PHONY: container_build container_tag acr_login container_push

ENGINE := docker
CONTAINER_NAME := pyrenew-hew
CONTAINER_REMOTE_NAME := $(ACR_TAG_PREFIX)$(CONTAINER_NAME)":latest"

container_build:
	$(ENGINE) build . -t $(CONTAINER_NAME)

container_tag:
	$(ENGINE) tag $(CONTAINER_NAME) $(CONTAINER_REMOTE_NAME)

acr_login:
	az acr login -n $(AZURE_CONTAINER_REGISTRY_ACCOUNT)

container_push: container_tag acr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)
