ARG TAG=latest

FROM cfaprdbatchcr.azurecr.io/pyrenew-hew-dependencies:${TAG}

COPY ./hewr /pyrenew-hew/hewr

WORKDIR /pyrenew-hew

COPY .ContainerBuildRprofile .Rprofile

RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epiprocess@main')"
RUN Rscript -e "pak::pkg_install('cmu-delphi/epipredict@main')"
RUN Rscript -e "pak::local_install('hewr')"

COPY . .
RUN pip install --root-user-action=ignore .
