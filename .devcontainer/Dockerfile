FROM python:3.6

COPY /.devcontainer/requirements.txt requirements.txt

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install git gcc unzip make libpq-dev libsndfile-dev -y \
    && pip install --no-cache-dir -r requirements.txt