#syntax=docker/dockerfile:1.7-labs

# -- R BASE IMAGE -- #

# We pin to a specific version for reproducibility; increment manually as needed
FROM rocker/tidyverse:4.5.2

# -- ARGS -- #

ARG GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

ARG GIT_BRANCH_NAME
ENV GIT_BRANCH_NAME=$GIT_BRANCH_NAME

ENV XLA_FLAGS=--xla_force_host_platform_device_count=4

# -- LANGUAGE BINARIES -- #

# Julia 1.11 from official image
COPY --from=julia:1.11 /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

# Copy in the binaries for UV Python - see https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# -- LOCAL DEPENDENCIES -- #

# R package - hewr
COPY ./hewr /cfa-stf-routine-forecasting/hewr

# Julia package - EpiAutoGP
COPY ./EpiAutoGP /cfa-stf-routine-forecasting/EpiAutoGP

WORKDIR /cfa-stf-routine-forecasting

# -- VENV MANAGEMENT AND DEPENDENCY SYNCING -- #

# R:
# Install pak and then the hewr package and its dependencies
RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "pak::local_install('hewr', upgrade = FALSE)"

# Julia:
# Cache Julia packages and artifacts
RUN --mount=type=cache,target=/root/.julia \
    julia --project=EpiAutoGP -e 'using Pkg; Pkg.instantiate()'

# Python:
# Some handy uv environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

# Copy in the UV/python dependency management configs
COPY ./pyproject.toml ./
COPY ./uv.lock ./
COPY ./.python-version ./

# Create the Pyrenew-Hew virtual environment and sync dependencies from pyproject.toml
RUN uv venv .venv
RUN --mount=type=cache,target=/root/.cache/uv

# Set VIRTUAL_ENV variable at runtime
ENV VIRTUAL_ENV=/pyrenew-hew/.venv

# Update PATH to use the selected venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Sync python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# -- PROJECT FILES -- #

COPY ./pyrenew_hew /cfa-stf-routine-forecasting/pyrenew_hew
COPY ./pipelines /cfa-stf-routine-forecasting/pipelines
COPY ./tests /cfa-stf-routine-forecasting/tests
COPY README.md /cfa-stf-routine-forecasting/README.md

# -- DAGSTER DEFINITIONS -- #
# Copy in the dagster definitions python file.
# All dagster definitions are defined here.
# Dagster Definitions are updated more frequently than other code,
# so it is much quicker for the build cache to worry about it last
COPY ./dagster_defs.py ./
