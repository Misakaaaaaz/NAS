#!/bin/bash



python hfai_train.py --device=0 --structure=6111 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &
python hfai_train.py --device=1 --structure=9930 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &
python hfai_train.py --device=2 --structure=3731 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &
python hfai_train.py --device=3 --structure=5111 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &
python hfai_train.py --device=4 --structure=81 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &
python hfai_train.py --device=5 --structure=1459 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &
python hfai_train.py --device=6 --structure=5292 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline &
python hfai_train.py --device=7 --structure=13539 --scheduler="cos" --batch_size=256 --seed=777 --wandb_mode=offline


# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name cifar10-calibrationdataset-test
