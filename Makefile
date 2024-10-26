.PHONY: container_push container_tag acr_login

CONTAINER_NAME := pyrenew-hew
CONTAINER_REMOTE_NAME := $(ACR_TAG_PREFIX)$(CONTAINER_NAME)":latest"

acr_login:
	az acr login -n $(AZURE_CONTAINER_REGISTRY_ACCOUNT)

container_tag:
	docker tag $(CONTAINER_NAME) $(CONTAINER_REMOTE_NAME)

container_push: acr_login container_tag
	docker push $(CONTAINER_REMOTE_NAME)

