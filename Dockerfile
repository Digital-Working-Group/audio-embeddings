FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y libgomp1 \
    libsndfile1 \
    libsndfile1-dev gcc \
    g++ \
    python3-dev \
    build-essential  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
WORKDIR /scripts
COPY ./requirements.txt .
COPY docker.env .env
RUN pip install -r requirements.txt

COPY ./embedding embedding
COPY ./utils utils
COPY __init__.py .
