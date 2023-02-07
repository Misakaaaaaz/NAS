#!/bin/bash



python hfai_train.py --device=0 --structure=6111 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &



# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name cifar10-calibrationdataset-test
