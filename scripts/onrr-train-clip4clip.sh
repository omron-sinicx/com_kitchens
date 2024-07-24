#!/bin/bash -eu

python -m com_kitchens.train experiment=onrr-clip4clip_en \
    trainer.max_epochs=10 \
    logger=csv
