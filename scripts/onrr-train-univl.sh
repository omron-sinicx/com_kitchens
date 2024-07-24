#!/bin/bash -eu

python -m com_kitchens.train experiment=onrr-univl_en \
    task_name=univl \
    trainer.max_epochs=10 \
    logger=csv
