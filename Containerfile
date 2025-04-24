#syntax=docker/dockerfile:1.7-labs

FROM python:3.13

ARG PYRENEW_VERSION="v0.1.2"
RUN apt-get update
RUN apt-get install -y r-base
RUN apt-get install -y cmake
RUN pip install --root-user-action=ignore -U pip
RUN pip install --root-user-action=ignore git+https://github.com/cdcgov/pyrenew.git@$PYRENEW_VERSION
RUN pip install --root-user-action=ignore quarto-cli

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
