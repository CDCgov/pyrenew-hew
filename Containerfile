#syntax=docker/dockerfile:1.7-labs
ARG TAG=latest

FROM cfaprdbatchcr.azurecr.io/pyrenew-hew-dependencies:${TAG}
ARG GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

ARG GIT_BRANCH_NAME
ENV GIT_BRANCH_NAME=$GIT_BRANCH_NAME

COPY ./hewr /pyrenew-hew/hewr

WORKDIR /pyrenew-hew

COPY .ContainerBuildRprofile .Rprofile

RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epiprocess@main')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epipredict@main')"
RUN Rscript -e "pak::local_install('hewr')"

COPY --exclude=pipelines/priors . .
RUN pip install --root-user-action=ignore .

# Copy priors folder last
COPY pipelines/priors pipelines/priors
