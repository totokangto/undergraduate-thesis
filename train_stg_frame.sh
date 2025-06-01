#!/bin/bash

output_path="/local_datasets/3dgs/output/salmon_stg2_spw50"
model_path="output_init/flame_salmon"
iters=15000

python train_stg_frames.py --read_config --config_path test/flame_steak_suite/cfg_args.json \
    -o ${output_path} -m ${model_path} -v /local_datasets/3dgs/flame_salmon \
    --first_load_iteration ${iters} --quiet --eval
