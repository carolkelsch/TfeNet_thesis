# !/bin/bash -e

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 12 \
    --start_epoch 31 --epoch 34 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 12 \
    --start_epoch 35 --epoch 38 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 12 \
    --start_epoch 39 --epoch 42 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 12 \
    --start_epoch 43 --epoch 46 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 12 \
    --start_epoch 47 --epoch 50 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 12 \
    --start_epoch 51 --epoch 54 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 12 \
    --start_epoch 55 --epoch 58 --sgd 1  --resumepart 0 --device 0 --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 1 --epoch 4 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' #\
    # --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 5 --epoch 8 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 9 --epoch 12 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 13 --epoch 16 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 17 --epoch 20 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 21 --epoch 24 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 25 --epoch 28 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'

sleep 5m

python my_main.py --model TfeNet -b 1 --save_dir TfeNet_CT_airways_small_new \
    --cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 8 \
    --start_epoch 29 --epoch 30 --sgd 1  --resumepart 0 --device 0 --small_airways --dataset_path '/home/carolinakelsch/Documents/ThesisDatasets/TfeNet_preprocessed/Dataset016_AirRCFold0' \
    --resume '/home/carolinakelsch/Documents/TfeNet_thesis/results/TfeNet_CT_airways_small_new/latest.ckpt'
