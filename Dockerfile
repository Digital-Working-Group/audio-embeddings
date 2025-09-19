FROM --platform=linux/amd64 pytorch/pytorch:latest
RUN apt-get update \
&& apt-get install -y \
libgl1-mesa-glx \
libx11-xcb1 \
&& apt-get clean all \
&& rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
astropy \
matplotlib \
pandas \
scikit-learn \
scikit-image 

RUN pip install torch
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./data2vec .
COPY ./utils /utils
COPY ./wav2vec .
WORKDIR /scripts