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

# install python 3.7
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.7 python3.7-dev python3.7-distutils

## set python3.9 as default
RUN cp /usr/bin/python3.7 /usr/bin/python3 && \
    cp /usr/bin/python3.7 /usr/bin/python
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

# install VidChapters
ADD VidChaptersEnhanced /app/VidChapters
RUN pip install -r /app/VidChapters/requirements.txt
# missing packages
RUN pip install pycocoevalcap==1.2 sentencepiece==0.1.99
# PyTorch
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

ENV PYTHONPATH /app/VidChapters
