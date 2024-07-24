#!/bin/bash -eu

mkdir -p data/vid2seq

# FrozenBiLM
mkdir -p data/frozenbilm
[ ! -f data/vid2seq/frozenbilm.csv ] && \
    python com_kitchens/preprocess/frozenbilm.py

docker run --rm -it --gpus all --shm-size=128g \
    -v $(realpath $(pwd)):/workspace \
    com_kitchens/frozenbilm \
    python /app/FrozenBiLM/extract/clip_video_features.py \
        --csv /workspace/data/vid2seq/frozenbilm.csv

# whisperX
mkdir -p data/whisperx
mkdir -p data/whisperx_align
[ ! -f data/vid2seq/whisperx.csv ] && \
    python com_kitchens/preprocess/whisperx.py

docker run --rm -it --gpus all --shm-size=128g \
    -v $(realpath $(pwd)):/workspace \
    com_kitchens/whisperx \
    python /workspace/scripts/whisperx_inference.py \
        --csv /workspace/data/vid2seq/whisperx.csv

docker run --rm -it --gpus all --shm-size=128g \
    -v $(realpath $(pwd)):/workspace \
    com_kitchens/whisperx \
    python /workspace/scripts/whisperx_merge_asr.py \
        /workspace/data/whisperx \
        /workspace/data/vid2seq/whisperx.pkl

docker run --rm -it --gpus all --shm-size=128g \
    -v $(realpath $(pwd)):/workspace \
    com_kitchens/whisperx \
    python /workspace/scripts/whisperx_align.py \
        --csv /workspace/data/vid2seq/whisperx.csv \
        --asr /workspace/data/vid2seq/whisperx.pkl \
        --output_path /workspace/data/whisperx_align \
        --model-dir /workspace/data/cache/

docker run --rm -it --gpus all --shm-size=128g \
    -v $(realpath $(pwd)):/workspace \
    com_kitchens/whisperx \
    python /workspace/scripts/whisperx_merge_asr_align.py \
        /workspace/data/whisperx_align \
        /workspace/data/vid2seq/asr.pkl \
        /workspace/data/vid2seq/whisperx.pkl

# vid2seq
mkdir -p data/vid2seq
docker run --rm -it --gpus all --shm-size=128g \
    -v $(realpath $(pwd)):/workspace \
    -v $(realpath $(pwd))/docker/VidChapters/VidChaptersEnhanced:/app/VidChapters \
    com_kitchens/vidchapters \
    python /app/VidChapters/preproc/comk.py

# download models
[ ! -f cache/vid2seq_htmchaptersyoucook.pth ] && gdown 1Kvx5OHJANtKVlyKe5oLvq6YOkewFqz8E -O cache/