# P100 llama.cpp patch bench

A/B test of the [Frostbite1536/P100](https://github.com/Frostbite1536/P100) llama.cpp patches on 4x Tesla P100-PCIE-16GB, against upstream llama.cpp and the [Kmic-68 llama.cpp fork](https://github.com/Kmic-68/llama.cpp), on 1, 2 and 4 cards. Run 2026-10-31.

## TLDR

- Patches 0001-0004 (quantized-KV conversion speedups, FA tile flush) gave **no measurable gain** on any test, on upstream or on Kmic.
- Patches 0005/0006 (MMVH half2 mat-vec) are a **large decode regression** on real P100s: -30% to -74% decode t/s. Prefill is unaffected.
- All patches pass the correctness tests (`test-backend-ops` MUL_MAT and FLASH_ATTN_EXT: 0 failures).
- Kmic-68 is the fastest build by far: on Qwen3.8-27B Q6, 2 cards, 26.2 vs 9.7 t/s decode and 371 vs 140 t/s prefill against upstream.
- Kmic + patches (MMVH off) runs identical to Kmic alone, so there is nothing to merge.
- Biggest real lever: **4 cards instead of 2** for Qwen3.8-27B with Kmic tensor split, +46% decode in llama-bench, +22-30% in llama-server with MTP.
- On 2 cards, the deployed llama-server settings are already within noise of the best tried; MTP roughly doubles decode (26 -> 42-52 t/s).

## Hardware and software

| | |
|---|---|
| GPUs | 4x Tesla P100-PCIE-16GB (sm_60), driver 580.178.04, P2P on (`GGML_CUDA_P2P=1`, `NCCL_P2P_LEVEL=SYS`) |
| Host | X79 board, 12 cores, 15 GB RAM |
| Toolchain | Docker image: Ubuntu 22.04, gcc 11.4, CUDA 12.9.1, NCCL 2.27.3 |
| Upstream | ggml-org/llama.cpp `79e2e74eb` (the commit the Frostbite patches are pinned to) |
| Frostbite | Frostbite1536/P100 `4b98b49`, `third_party/patches/0001-0006` |
| Kmic | Kmic-68/llama.cpp `ae35056eb` (the production binary) |

Build flags (all arms): `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`. See `scripts/dbuild.sh`.

### Arms

| Arm | Build | Notes |
|---|---|---|
| `up` | upstream at pin | baseline |
| `pat` | upstream + 0001-0006 | MMVH on (default on cc 6.0) |
| `patnommvh` | upstream + 0001-0006 | `GGML_CUDA_DISABLE_MMVH=1` |
| `kmic` | Kmic-68 | production binary |
| `kf` | Kmic + 0001-0003, 0005, 0006 | MMVH on; 0004 not applied |
| `kfnommvh` | same as `kf` | `GGML_CUDA_DISABLE_MMVH=1` |

## Method

- `llama-bench -fa 1 -ctk q4_0 -ctv q4_0 -r 3 -p 512 -n 128 -d 0,16384`, `-ngl 99`.
- Multi-card: Kmic arms use `-sm tensor`. Upstream arms use `-sm layer`; tensor split was not tested on upstream.
- Arms are interleaved and the order rotates each round; 3 rounds, so each cell is the median of 3 rounds x 3 reps.
- Cool-down to <= 48 C before every run; nvidia-smi logged at 1 Hz (`results/*/gpu.csv`); watchdog kills a run above 80 C.
- All case fans at 100% PWM during the tests.
- Production services were stopped for the whole run, so nothing else was on the cards.

Scripts: `scripts/ab2.sh` (bench), `scripts/queue2.sh` (phases), `scripts/sum.py` (summary).

## Results: llama-bench (t/s, median of 3 rounds)

### Phase 1: 1 card

**Qwen3-VL-4B-Instruct UD-Q4_K_XL**

| test | up | pat | patnommvh | kmic | kf | kfnommvh |
|---|---|---|---|---|---|---|
| pp512 | 1018.6 | 1013.1 | 1017.9 | **1235.8** | 1230.4 | 1240.6 |
| pp512 @16k | 328.3 | 325.7 | 326.5 | **587.9** | 586.2 | 587.8 |
| tg128 | 57.0 | 23.8 | 57.0 | **73.1** | 23.3 | 73.1 |
| tg128 @16k | 31.6 | 18.2 | 31.6 | **45.9** | 19.9 | 45.9 |

**Qwen3.6-35B-A3B UD-Q2_K_XL (MoE)**

| test | up | pat | patnommvh | kmic | kf | kfnommvh |
|---|---|---|---|---|---|---|
| pp512 | 512.4 | 510.4 | 510.5 | **569.9** | 568.0 | 569.3 |
| pp512 @16k | 390.5 | 388.1 | 389.5 | **505.9** | 504.3 | 505.6 |
| tg128 | 49.0 | 34.3 | 48.9 | **59.3** | 36.0 | 59.3 |
| tg128 @16k | 42.8 | 31.1 | 42.8 | crash | crash | crash |

### Phase 2: 2 cards

**Qwen3.8-27B UD-Q6_K_XL** (the production model)

| test | up (layer) | patnommvh (layer) | kmic (tensor) | kf (tensor) | kfnommvh (tensor) |
|---|---|---|---|---|---|
| pp512 | 139.8 | 143.7 | **371.2** | 370.6 | 353.6* |
| pp512 @16k | 124.3 | 123.7 | **349.9** | 347.9 | 349.9 |
| tg128 | 9.7 | 9.7 | **26.2** | 6.9 | 26.2 |
| tg128 @16k | 9.0 | 9.0 | **25.4** | 6.9 | 25.4 |

\* one slow sample (269.6); the other two were 373.

**Qwen3.6-35B-A3B UD-Q4_K_XL (MoE)**

| test | up (layer) | patnommvh (layer) | kmic (tensor) | kf (tensor) | kfnommvh (tensor) |
|---|---|---|---|---|---|
| pp512 | 488.8 | 488.5 | **839.2** | 838.7 | 839.7 |
| pp512 @16k | 376.4 | 375.6 | **754.8** | 753.4 | 754.0 |
| tg128 | 55.5 | 55.3 | **66.9** | 58.5 | 67.0 |
| tg128 @16k | 47.7 | 47.7 | crash | crash | crash |

### Phase 3: 4 cards

**Qwen3.8-27B UD-Q6_K_XL**

| test | up (layer) | patnommvh (layer) | kmic (tensor) | kfnommvh (tensor) |
|---|---|---|---|---|
| pp512 | 144.3 | 142.5 | **508.7** | 508.3 |
| pp512 @16k | 125.3 | 124.8 | **511.0** | 511.2 |
| tg128 | 9.7 | 9.7 | **38.3** | 38.3 |
| tg128 @16k | 9.0 | 9.0 | **39.0** | 39.0 |

The `pat` and `kf` arms (MMVH on) were left out of phase 3 because phases 1 and 2 already showed the regression.

### Kmic scaling, Qwen3.8-27B Q6

| cards | tg128 | pp512 | pp512 @16k |
|---|---|---|---|
| 2 | 26.2 | 371 | 350 |
| 4 | 38.3 (+46%) | 509 (+37%) | 511 (+46%) |

## Results: llama-server (Kmic, Qwen3.8-27B Q6, real requests)

`scripts/srvbench.py`. Each config: fresh server, cool-down, one discarded warmup request, then a ~2k-token prose summarize prompt and a short code-generation prompt. Each request uses `max_tokens 512`, no prompt cache and thinking off. 3 rounds, interleaved. Base args match production: `-c 262144 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0`, MTP draft with q4_0 draft KV, `--temp 1.0 --top-k 20 --top-p 0.95`.

| config | cards | changes vs production | code tg | prose tg | prose pp | draft accept (code/prose) |
|---|---|---|---|---|---|---|
| prod2 | 2 | as deployed: `-b 2048 -ub 2048`, MTP n-max 3, p-min 0.0 | 50.1 | 42.5 | 386.5 | 0.80 / 0.62 |
| quick2 | 2 | Kmic QUICKSTART: `-b 32768`, n-max 4, p-min 0.2 | 52.1 | 42.5 | 386.6 | 0.75 / 0.57 |
| n4p0_2 | 2 | n-max 4 | 50.1 | 42.6 | 386.0 | 0.71 / 0.56 |
| n3p02_2 | 2 | p-min 0.2 | 51.1 | 43.3 | 386.3 | 0.82 / 0.65 |
| b32k_2 | 2 | `-b 32768` | 48.5 | 44.4 | 386.9 | 0.76 / 0.67 |
| kf2 | 2 | Kmic + patches, MMVH off | 49.2 | 42.0 | 385.8 | 0.77 / 0.61 |
| nomtp2 | 2 | MTP off | 26.1 | 26.2 | 401.8 | - |
| prod4 | 4 | production settings on 4 cards | 61.5 | 51.8 | 532.2 | 0.82 / 0.63 |
| **quick4** | 4 | QUICKSTART settings on 4 cards | **64.9** | **52.8** | 530.6 | 0.76 / 0.58 |

Values are medians of 3. The code prompt's tg varies run to run by up to about 10% because MTP acceptance differs at temperature 1.0, so differences under about 3 t/s between 2-card configs are noise.

## Correctness

`test-backend-ops` on GPU 0 (`scripts/correct.sh`, `scripts/correct2.sh`):

| arm | MUL_MAT | FLASH_ATTN_EXT |
|---|---|---|
| up | 2423/2423 pass | 3993/3993 pass |
| pat | 2481/2481 pass | 4009/4009 pass (+16/16 biased-V cases from 0003) |
| patnommvh | 2481/2481 pass | - |
| kf | 1418/1418 pass | 4035/4035 pass |
| kfnommvh | 1418/1418 pass | - |

The full MUL_MAT run aborts on **every** arm, upstream included, at `ggml-cuda.cu:1463 GGML_ASSERT(convert_func != nullptr)`, on non-contiguous K-quant src0 cases (`k_v > 0`). The abort is in upstream and is not caused by the patches. 1208 cases had passed with 0 failures before the abort. The MUL_MAT counts above come from a rerun that skips those cases with the filter in `correct2.sh`.

## Findings

1. **MMVH (0005/0006) is a regression on P100 hardware.** It cuts decode by 59% on the 4B model on 1 card, 30% on the 35B MoE, and 74% on Qwen3.8-27B Q6 on 2 cards. The same happens on upstream and on Kmic. Prefill is unaffected because MMVH only covers batch 1-8. Whether the kernel itself is slow or the dispatch picks it in bad cases was **not** isolated (no per-kernel profile was taken). The patch author should check this on real sm_60 hardware.
2. **0001/0002/0004 (quantized-KV conversion, FA tile flush) give no gain.** With q4_0 KV at 16k depth, prefill and decode stay within run-to-run noise of the unpatched build.
3. **Kmic + patches = Kmic.** Kmic's own decode path (`mmvq-f16`) and FA path already cover what these patches target.
4. **Card count is the lever.** Kmic tensor split scales from 2 to 4 cards at +46% decode in llama-bench and +22-30% in llama-server.
5. **2-card server flags are at their limit.** None of the batch, MTP n-max or p-min variants beat production by more than noise. MTP is worth about 2x decode and must stay on.

## Known issue: Kmic crash on MoE decode at depth

Every Kmic-based arm (`kmic`, `kf`, `kfnommvh`) aborted with `ggml-cuda.cu:112: CUDA error` on the **Qwen3.6-35B-A3B** tg128 test at 16k depth. This happened on 1 card (Q2_K_XL) and on 2 cards with tensor split (Q4_K_XL), in all 3 rounds. Upstream ran the same tests without error. Dense Qwen3.8-27B never hit it. On 2 cards the Q4 model has plenty of free memory, so out-of-memory is unlikely; a Kmic bug in that MoE path is the more likely cause (**inferred, not confirmed**). Logs: `results/p1/kmic.Qwen3.6-35B-A3B-*.err`, `results/p2/kmic.Qwen3.6-35B-A3B-*.err`.

## Measured vs inferred

- **Measured:** every t/s number, the draft acceptance rates, the correctness counts, temperatures, and that the crash happens.
- **Inferred:** why MMVH is slow; why the Kmic MoE crash happens; that the 0001/0002/0004 code paths overlap with what Kmic already does.

## Thermals

Peak GPU temperature: 61 C (phase 1), 58 C (phase 2), 54 C (phase 3), 54 C (server tests). No watchdog aborts. Every run started at <= 48 C. One case fan maxes out at about 1550 RPM even at 100% PWM; the other two ran at 2250 and 3000 RPM.

## Layout

```
README.md          this report
scripts/           build, bench, correctness and server-bench scripts
results/p1..p3/    llama-bench jsonl + stderr per arm/model/round, 1 Hz gpu.csv, run log
results/srv/       llama-server results.jsonl, per-config server logs, run log
results/correct*/  test-backend-ops output
```

Paths in scripts and logs have been anonymized.
