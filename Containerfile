FROM python:3.12

RUN apt-get update
RUN apt-get install -y r-base
RUN apt-get install -y cmake
RUN pip install --root-user-action=ignore -U pip 
RUN pip install --root-user-action=ignore git+https://github.com/cdcgov/pyrenew
COPY ./hewr ./pyrenew-hew/hewr

WORKDIR pyrenew-hew

COPY .ContainerBuildRprofile .Rprofile

RUN Rscript -e "install.packages('pak')"
RUN Rscript -e "pak::local_install('hewr')"

COPY . .

RUN pip install --root-user-action=ignore .

