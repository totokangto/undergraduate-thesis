#!/bin/bash

output_path="/local_datasets/3dgs/output/coffee_spw50"
model_path="output_init/coffee_martini"
iters=15000

python train_frames.py --read_config --config_path test/flame_steak_suite/cfg_args.json \
    -o ${output_path} -m ${model_path} -v /local_datasets/3dgs/coffee_martini \
    --first_load_iteration ${iters} --quiet 