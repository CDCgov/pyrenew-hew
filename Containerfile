FROM python:3.12

RUN apt-get update
RUN apt-get install -y r-base

COPY . ./pyrenew-hew

WORKDIR pyrenew-hew

COPY .ContainerBuildRprofile .Rprofile

RUN pip install --root-user-action=ignore .
