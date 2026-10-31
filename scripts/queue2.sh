#!/bin/bash
B=/data/p100ab; cd $B
M4B=/home/user/models/qwen3-vl-4b/Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf
M36Q2=/data/models/Qwen3.6-35B-A3B-MTP-UD-Q2_K_XL.gguf
M38Q6=/home/user/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
M36Q4=/data/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
echo "$(date +%T) phase1" >> queue.log
OUT=$B/res/p1 GPUS=0 MODELS="$M4B $M36Q2" ARMS="up pat patnommvh kmic kf kfnommvh" ROUNDS=3 ./ab2.sh
echo "$(date +%T) phase2" >> queue.log
OUT=$B/res/p2 GPUS=0,1 SM=tensor SMU=layer MODELS="$M38Q6 $M36Q4" ARMS="up patnommvh kmic kfnommvh kf" ROUNDS=3 ./ab2.sh
echo "$(date +%T) phase3" >> queue.log
OUT=$B/res/p3 GPUS=0,1,2,3 SM=tensor SMU=layer MODELS="$M38Q6" ARMS="up patnommvh kmic kfnommvh" ROUNDS=3 ./ab2.sh
echo "$(date +%T) QUEUE-DONE" >> queue.log
