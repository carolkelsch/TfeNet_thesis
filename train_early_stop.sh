# !/bin/bash -e

python my_main.py --model TfeNet -b 1 --save_dir Dataset016_AirRC/TfeNet \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --epoch 2 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRC' \
    --early_stop