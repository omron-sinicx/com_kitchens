#!/bin/sh
#$-l rt_C.large=1
#$-j y

python -m com_kitchens.preprocess.video -i ./data/main -o ./data/frames --cpu 80
