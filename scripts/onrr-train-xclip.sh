#!/bin/bash -eu

python -m com_kitchens.train experiment=onrr-xclip_en \
    task_name=xclip \
    trainer.max_epochs=10 \
    logger=csv
