#syntax=docker/dockerfile:1-labs

FROM rocker/tidyverse:4.5.1

ARG GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

ARG GIT_BRANCH_NAME
ENV GIT_BRANCH_NAME=$GIT_BRANCH_NAME

ENV XLA_FLAGS=--xla_force_host_platform_device_count=4

COPY ./hewr /pyrenew-hew/hewr
WORKDIR /pyrenew-hew

# install hewr dependencies
RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "pak::local_install('hewr', upgrade = FALSE)"

#
# Python from https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Some handy uv environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

# copy in the project files
COPY ./pyrenew_hew ./pyrenew_hew
COPY ./pipelines ./pipelines
COPY ./tests ./tests
COPY README.md ./
COPY ./pyproject.toml ./
COPY ./uv.lock ./
COPY ./.python-version ./

# VENV MANAGEMENT AND DEPENDENCY SYNCING

# Create the Pyrenew-Hew venv and sync dependencies from pyproject.toml
RUN uv venv .venv
RUN --mount=type=cache,target=/root/.cache/uv
# Set VIRTUAL_ENV variable at runtime to choose which venv to activate
# By default we'll do the main pyrenew-hew venv
ENV VIRTUAL_ENV=/pyrenew-hew/.venv
RUN uv sync

# Copy in the dagster defs
COPY ./dagster_defs.py ./

# Update PATH to use the selected venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
