#syntax=docker/dockerfile:1-labs

FROM rocker/tidyverse:4.4.3@sha256:da14abcd1ffa4e63093aba87a23a5cffc364e7db8f770c194965519743339760

#
# General Build Args and Environment Variables
#

ARG GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA
ARG GIT_BRANCH_NAME
ENV GIT_BRANCH_NAME=$GIT_BRANCH_NAME
ENV XLA_FLAGS=--xla_force_host_platform_device_count=4

#
# Additional programming language compilers/interpreters
#

# Julia 1.11 from official image
COPY --from=julia:1.11.9@sha256:72f6e546aa752c833da71a20c57c70e73f3f41016a9ba2dfffeac84b9fbb44d7 /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

# Python from https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/

# Some handy uv environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

#
# Copy local dependencies into our container, then install them
#

# R package - hewr
COPY ./hewr /cfa-stf-routine-forecasting/hewr

# Julia package - EpiAutoGP
COPY ./EpiAutoGP /cfa-stf-routine-forecasting/EpiAutoGP

# Set working directory
WORKDIR /cfa-stf-routine-forecasting

# Install hewr
RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "pak::local_install('hewr', upgrade = FALSE)"

# Cache Julia packages and artifacts
RUN --mount=type=cache,target=/root/.julia \
    julia --project=EpiAutoGP -e 'using Pkg; Pkg.instantiate()'

#
# Bring in python project dependency information and set the virtual env
#

# Dependency information
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
COPY README.md ./README.md

# Set VIRTUAL_ENV variable at runtime
ENV VIRTUAL_ENV=/cfa-stf-routine-forecasting/.venv

# Create the virtual environment
RUN uv venv "${VIRTUAL_ENV}"

# Update PATH to use the selected venv at runtime
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Sync all python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

#
# Copy in python pipeline and orchestration files that frequently change
#

# Project files
COPY pipelines ./pipelines

# Dagster
COPY dagster_defs.py ./dagster_defs.py
