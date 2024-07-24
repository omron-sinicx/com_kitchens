#!/bin/bash -eu

docker run --rm -it --gpus all --shm-size=128g \
    -v $(realpath $(pwd)):/workspace \
    com_kitchens/vidchapters \
    python -m torch.distributed.launch \
        --nproc_per_node 4 \
        --use_env /app/VidChapters/dvc.py \
        --epochs 40 \
        --lr 3e-4 \
        --save_dir comk_zs \
        --combine_datasets comk \
        --combine_datasets_val comk \
        --train_json_path /workspace/data/vid2seq/train.json \
        --val_json_path /workspace/data/vid2seq/val.json \
        --test_json_path /workspace/data/vid2seq/val.json \
        --features_path /workspace/data/frozenbilm \
        --subtitles_path /workspace/data/vid2seq/asr_proc.pkl \
        --presave_dir /workspace/logs/vid2seq \
        --batch_size 4 \
        --batch_size_val 4 \
        --schedule="cosine_with_warmup" \
        --load /workspace/cache/vid2seq_htmchaptersyoucook.pth \
        --eval
