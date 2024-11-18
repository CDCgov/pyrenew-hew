.PHONY: container_build container_tag acr_login container_push dep_container_build dep_container_tag

ENGINE := docker
DEP_CONTAINER_NAME := pyrenew-hew-dependencies
DEP_CONTAINERFILE := Containerfile.dependencies
DEP_CONTAINER_REMOTE_NAME := $(ACR_TAG_PREFIX)$(DEP_CONTAINER_NAME):latest
CONTAINER_NAME := pyrenew-hew
CONTAINERFILE := Containerfile
CONTAINER_REMOTE_NAME := $(ACR_TAG_PREFIX)$(CONTAINER_NAME):latest

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
	$(ENGINE) push $(DEP_CONTAINER_NAME)

container_push: container_tag acr_login
	$(ENGINE) push $(CONTAINER_REMOTE_NAME)
