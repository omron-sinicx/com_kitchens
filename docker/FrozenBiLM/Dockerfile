# Host the FrozenBiLM repository for extracting CLIP features
# https://github.com/antoyang/FrozenBiLM

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

## avoid interaction on installing software-properties-common
ENV TZ Asia/Tokyo
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# update/install packages
RUN rm /etc/apt/sources.list.d/*
RUN apt-get update && apt-get -y install \
    python3 python3-pip python3-dev git \
    wget curl ffmpeg default-jre

# install python 3.9
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.9 python3.9-dev

## set python3.9 as default
RUN cp /usr/bin/python3.9 /usr/bin/python3 && \
    cp /usr/bin/python3.9 /usr/bin/python
RUN pip3 install --upgrade pip
RUN pip3 install setuptools

# remove cache
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# workspace & app directories
WORKDIR /workspace
RUN mkdir -p /app
RUN chmod 777 -R /workspace /app

# add hostuser
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" && \
  useradd -l -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}"

USER $USER_NAME
ENV PATH $PATH:/home/${USER_NAME}/.local/bin

# Install FrozenBiLM
RUN git clone https://github.com/antoyang/FrozenBiLM.git /app/FrozenBiLM
RUN cd /app/FrozenBiLM && git checkout fd7d44126de47a6e9e0d9e459d07191b466f3cea && cd -

RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r /app/FrozenBiLM/requirements.txt

## avoid naming conflict
RUN mv /app/FrozenBiLM/extract/extract.py /app/FrozenBiLM/extract/clip_video_features.py

## set environment variable
ENV PYTHONPATH /app/FrozenBiLM

# Install CLIP
RUN git clone https://github.com/openai/CLIP.git /app/CLIP
RUN cd /app/CLIP && git checkout a1d071733d7111c9c014f024669f959182114e33  && cd -

RUN pip install ftfy regex tqdm
RUN pip install -e /app/CLIP

## fetch ViT-L/14 to avoid downloading multiple times
RUN python -c "import clip; clip.load('ViT-L/14');"

# ffmpeg
RUN pip install ffmpeg ffmpeg-python
