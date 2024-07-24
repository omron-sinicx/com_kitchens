# WhisperX v3.0.1 https://github.com/m-bain/whisperX/tree/v3.0.1
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

# install python 3.10
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.10 python3.10-dev python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

## set python3.10 as default
RUN cp /usr/bin/python3.10 /usr/bin/python3 && \
    cp /usr/bin/python3.10 /usr/bin/python
RUN pip3 install setuptools
# pip>=24.1 does not accept metadata of pyannote.audio==2.1.1
RUN pip3 install --upgrade pip==23.3.2

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

# whisperx
RUN pip install whisperx==3.1.5 --no-deps
# requirements.txt
RUN pip install \
    torch==1.12.1+cu113 \
    torchaudio==0.12.1 \
    faster-whisper==1.0.0 \
    transformers==4.31.0 \
    pandas==2.2.2 \
    nltk==3.8.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113
# missing packages
RUN pip install \
    pyannote.audio==2.1.1

# download Whisper Large-V2 model
RUN python -c "import whisperx; whisperx.load_model('large-v2', compute_type='int8', device='cpu', language='en', asr_options={'repetition_penalty': 1,'prompt_reset_on_temperature': 0.5,'no_repeat_ngram_size': 2});"
RUN python -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu');"

# overwrite some packages for CUDA11.3
RUN pip install ctranslate2==3.20.0