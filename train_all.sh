# !/bin/bash -e

python my_main.py --model TfeNet -b 1 --save-dir TfeNet_CT_airways_new \
--cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 4 \
--start-epoch 1 --epoch 2 --sgd 1  --resumepart 0 --device 0


python my_main.py --model TfeNet -b 1 --save-dir TfeNet_CT_airways_small_new \
--cubesize 128 128 128 --stridet 64 64 64 --stridev 64 64 64 --worker 4 \
--start-epoch 1 --epoch 2 --sgd 1  --resumepart 0 --device 0 --small_airways

