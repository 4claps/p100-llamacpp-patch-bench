#!/bin/bash
# MUL_MAT correctness skipping the known pre-existing abort (non-contiguous K-quant src0, k_v>0)
B=/data/p100ab; OUT=$B/res/correct2; mkdir -p $OUT
F="^(?!.*(src1_scale=|type_a=(q[2-6]_K|iq[0-9a-z_]+).*k_v=[1-9]))"
for a in pat:pat:"" patnommvh:pat:"-e GGML_CUDA_DISABLE_MMVH=1" up:up:"" kf:kmicfrost:"" kfnommvh:kmicfrost:"-e GGML_CUDA_DISABLE_MMVH=1"; do
 IFS=: read n t e <<< "$a"
 timeout 5400 docker run --rm --name abrun --gpus "\"device=0\"" $e -v $B:$B --entrypoint $B/$t/build/bin/test-backend-ops p100-llamacpp-test:latest test -b CUDA0 -o MUL_MAT -p "$F" > $OUT/$n.MUL_MAT.txt 2>&1
 echo "$n rc=$? $(grep -E "tests passed" $OUT/$n.MUL_MAT.txt | tail -1) fails=$(grep -c FAIL $OUT/$n.MUL_MAT.txt)" >> $OUT/summary
done
for n in kf; do timeout 3600 docker run --rm --name abrun --gpus "\"device=0\"" -v $B:$B --entrypoint $B/kmicfrost/build/bin/test-backend-ops p100-llamacpp-test:latest test -b CUDA0 -o FLASH_ATTN_EXT > $OUT/kf.FA.txt 2>&1; echo "kf FA rc=$? $(grep -E "tests passed" $OUT/kf.FA.txt | tail -1)" >> $OUT/summary; done
echo DONE >> $OUT/summary
