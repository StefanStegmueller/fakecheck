FROM python:3.8

#Install requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /src
