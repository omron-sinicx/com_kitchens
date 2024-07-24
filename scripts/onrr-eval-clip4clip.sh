#!/bin/bash -eu

CKPT_PATH=${1:-}

if [ -z $CKPT_PATH ]; then
    echo "Usage: ${0} <ckpt_path>"
    exit 1
fi

python -m com_kitchens.eval task_name=eval-clip4clip-FEASIBLE \
    data=comkitchens_xclip_en.yaml \
    data.dat_files.test=${path/to/your/dat_file} \
    data.test_frame_ratios=[0.25] \
    model.task_config.stage=early \
    model=clip4clip.yaml \
    trainer=ddp.yaml \
    trainer.gpus=1 \
    ckpt_path=$CKPT_PATH

python -m com_kitchens.eval task_name=eval-clip4clip-STAGE \
    data=comkitchens_xclip_en.yaml \
    data.test_frame_ratios=[0.25] \
    model.task_config.stage=early \
    model=clip4clip.yaml \
    trainer=ddp.yaml \
    trainer.gpus=1 \
    ckpt_path=$CKPT_PATH
