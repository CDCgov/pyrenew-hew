FROM python:3.12

COPY . ./pyrenew_hew

RUN pip install --root-user-action=ignore ./pyrenew_hew
