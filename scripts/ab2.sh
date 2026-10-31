#!/bin/bash
# Interleaved A/B in the prod image.
# env: GPUS ("0" or "0,1"), MODELS, ARMS, CONFIGS (kv types), SM (none|layer|tensor for kmic arms),
#      SMU (split mode for upstream arms), ROUNDS, OUT, PD (prompt/depth args), EXTRA
set -u
B=/data/p100ab; OUT=${OUT:?}; mkdir -p "$OUT"
ROUNDS=${ROUNDS:-3}; GPUS=${GPUS:-0}; SM=${SM:-none}; SMU=${SMU:-layer}
PD=${PD:-"-p 512 -n 128 -d 0,16384"}; EXTRA=${EXTRA:-}
ARMS=${ARMS:-"up pat patnommvh"}; CONFIGS=${CONFIGS:-"q4_0"}

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used,fan.speed \
  --format=csv -l 1 > "$OUT/gpu.csv" 2>&1 &
SMI=$!
( while sleep 2; do
    wt=$(nvidia-smi -i "$GPUS" --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)
    if [ "$wt" -gt 80 ]; then echo "$(date +%T) ABORT temp $wt" >> "$OUT/log"; docker kill abrun >/dev/null 2>&1; fi
  done ) &
WD=$!
trap 'kill $SMI $WD 2>/dev/null' EXIT

cool() {
  for i in $(seq 1 120); do
    local tc; tc=$(nvidia-smi -i "$GPUS" --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)
    [ "$tc" -le 48 ] && return
    sleep 5
  done
  echo "cool-timeout $tc" >> "$OUT/log"
}
bin_of() {
  case $1 in
    up) echo $B/up/build/bin ;;
    pat|patnommvh) echo $B/pat/build/bin ;;
    kmic) echo /workspace/build-nccl/bin ;;
    kf|kfnommvh|kf*) echo $B/kmicfrost/build/bin ;;
  esac
}
env_of() { case $1 in *nommvh) echo "-e GGML_CUDA_DISABLE_MMVH=1" ;; esac; }
smarg() {
  [ "$SM" = none ] && return
  case $1 in up|pat|patnommvh) echo "-sm $SMU" ;; *) echo "-sm $SM" ;; esac
}

for r in $(seq 1 "$ROUNDS"); do
  for m in $MODELS; do
    mn=$(basename "$m" .gguf)
    for kv in $CONFIGS; do
      arms=($ARMS); n=${#arms[@]}
      for i in $(seq 0 $((n-1))); do
        a=${arms[$(( (i+r) % n ))]}
        cool
        echo "$(date +%T) r$r $mn $kv $a" >> "$OUT/log"
        timeout 2400 docker run --rm --name abrun --gpus "\"device=$GPUS\"" \
          -e GGML_CUDA_P2P=1 -e GGML_CUDA_GRAPHS_PRE_VOLTA=3 -e NCCL_P2P_LEVEL=SYS $(env_of "$a") \
          -v $B:$B -v /home/user/p100-docker-test/build-nccl:/workspace/build-nccl:ro \
          -v /data/models:/data/models:ro -v /home/user/models:/home/user/models:ro \
          --entrypoint "$(bin_of "$a")/llama-bench" p100-llamacpp-test:latest \
          -m "$m" -ngl 99 -fa 1 -ctk "$kv" -ctv "$kv" -r 3 -o jsonl $PD $(smarg "$a") $EXTRA \
          > "$OUT/$a.$mn.$kv.r$r.jsonl" 2> "$OUT/$a.$mn.$kv.r$r.err" \
          || { echo "  rc=$? $a $mn $kv r$r" >> "$OUT/log"; docker kill abrun >/dev/null 2>&1; }
      done
    done
  done
done
echo AB-DONE >> "$OUT/log"
