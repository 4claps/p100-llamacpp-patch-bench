#!/bin/bash
# build trees given as args inside the prod toolchain image
for t in "$@"; do
 docker run --rm --gpus all --user $(id -u):$(id -g) -v /data/p100ab:/data/p100ab -w /data/p100ab p100-llamacpp-test:latest bash -c "
  cmake -S $t -B $t/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON > $t.cmake.log 2>&1 && ninja -C $t/build -j10 llama-bench test-backend-ops llama-perplexity llama-server > $t.ninja.log 2>&1; echo BUILD $t rc=\$?"
done
