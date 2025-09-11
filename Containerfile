#syntax=docker/dockerfile:1.7-labs

FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

ARG GIT_BRANCH_NAME
ENV GIT_BRANCH_NAME=$GIT_BRANCH_NAME

# Some handy uv environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ENV XLA_FLAGS=--xla_force_host_platform_device_count=4


# R
RUN apt-get update
RUN apt-get install -y r-base
RUN apt-get install -y cmake

COPY ./hewr /pyrenew-hew/hewr

WORKDIR /pyrenew-hew

COPY .ContainerBuildRprofile .Rprofile

RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "install.packages('remotes')"
RUN Rscript -e "remotes::install_version('feasts', '0.4.2')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epiprocess@main')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epipredict@main')"
RUN Rscript -e "pak::local_install('hewr', upgrade = FALSE)"


COPY --exclude=pipelines/priors . .

# Python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync
RUN uv add quarto-cli

# Copy priors folder last
COPY pipelines/priors pipelines/priors
