#!/bin/bash
# correctness on card 0: MUL_MAT (pat, pat-nommvh, up) and FLASH_ATTN_EXT (pat, up)
B=/data/p100ab; OUT=$B/res/correct; mkdir -p $OUT
t() { name=$1; shift; docker run --rm --name abrun --gpus "\"device=0\"" "$@" -v $B:$B p100-llamacpp-test:latest 2>&1 ; }
for a in pat up; do
 for op in MUL_MAT FLASH_ATTN_EXT; do
  timeout 3600 docker run --rm --name abrun --gpus "\"device=0\"" -v $B:$B --entrypoint $B/$a/build/bin/test-backend-ops p100-llamacpp-test:latest test -b CUDA0 -o $op > $OUT/$a.$op.txt 2>&1; echo "$a $op rc=$? $(grep -E "tests passed|Backend .*: (OK|FAIL)" $OUT/$a.$op.txt | tail -2 | tr "\n" " ")" >> $OUT/summary
 done
done
timeout 3600 docker run --rm --name abrun --gpus "\"device=0\"" -e GGML_CUDA_DISABLE_MMVH=1 -v $B:$B --entrypoint $B/pat/build/bin/test-backend-ops p100-llamacpp-test:latest test -b CUDA0 -o MUL_MAT > $OUT/patnommvh.MUL_MAT.txt 2>&1; echo "patnommvh MUL_MAT rc=$? $(grep -E "tests passed" $OUT/patnommvh.MUL_MAT.txt | tail -1)" >> $OUT/summary
timeout 1800 docker run --rm --name abrun --gpus "\"device=0\"" -v $B:$B --entrypoint $B/pat/build/bin/test-backend-ops p100-llamacpp-test:latest test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias > $OUT/pat.vbias.txt 2>&1; echo "pat vbias rc=$? $(grep -E "tests passed" $OUT/pat.vbias.txt | tail -1)" >> $OUT/summary
timeout 1800 docker run --rm --name abrun --gpus "\"device=0\"" -v $B:$B --entrypoint $B/up/build/bin/test-backend-ops p100-llamacpp-test:latest test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias > $OUT/up.vbias.txt 2>&1; echo "up vbias rc=$? $(grep -E "tests passed" $OUT/up.vbias.txt | tail -1)" >> $OUT/summary
echo CORRECT-DONE >> $OUT/summary
