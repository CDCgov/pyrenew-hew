#syntax=docker/dockerfile:1-labs

FROM rocker/tidyverse

ARG GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

ARG GIT_BRANCH_NAME
ENV GIT_BRANCH_NAME=$GIT_BRANCH_NAME

ENV XLA_FLAGS=--xla_force_host_platform_device_count=4

COPY ./hewr /pyrenew-hew/hewr

WORKDIR /pyrenew-hew

RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epiprocess@main')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epipredict@main')"
RUN Rscript -e "pak::local_install('hewr', upgrade = FALSE)"

COPY --exclude=pipelines/priors . .
COPY pipelines/priors pipelines/priors

# Python from https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Some handy uv environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# copy in the dagster workflow definitions
COPY ./dagster_defs.py .

# create a virtual environment for the dagster workflows
ARG VIRTUAL_ENV=/pyrenew-hew/.dg_venv
RUN uv venv ${VIRTUAL_ENV}

# install the dagster workflow dependencies
RUN uv sync --script ./dagster_defs.py --active
# add the dagster workflow dependencies to the system path
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"