#!/usr/bin/env python3
"""Server-level A/B: start llama-server configs in the prod image, send fixed prompts, record timings.

usage: srvbench.py OUTDIR ROUNDS CONFIG_NAME... (configs defined in CONFIGS below)
Interleaves configs per round; cool-down to <=48C before each server start; discards a warmup request.
"""
import json, os, subprocess, sys, time, urllib.request

B = "/data/p100ab"
MODEL = "/home/user/models/Qwen3.8-27B-UD-Q6_K_XL.gguf"
WIKI = "/data/models/ab/wiki.test.raw"

BASE = ("--host 0.0.0.0 --port 8099 -m {model} --jinja --cache-ram 0 --no-cache-idle-slots --parallel 1 "
        "-c 262144 -n 25000 -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -lm none -fit off "
        "-ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0")
PROD_ENV = {"GGML_CUDA_P2P": "1", "GGML_CUDA_GRAPHS_PRE_VOLTA": "3", "LLAMA_SPEC_SAMPLE_TEMP": "1.0",
            "LLAMA_SPEC_DRAFT_TOPK": "20", "NCCL_P2P_LEVEL": "SYS"}

def cfg(gpus, ts, extra, binary="kmic", env=None):
    return {"gpus": gpus, "args": BASE + f" -ts {ts} " + extra, "bin": binary, "env": {**PROD_ENV, **(env or {})}}

CONFIGS = {
    # production as deployed (2 cards)
    "prod2":      cfg("0,1", "1/1", "-b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0"),
    # Kmic QUICKSTART recommendation
    "quick2":     cfg("0,1", "1/1", "-b 32768 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2"),
    "n4p0_2":     cfg("0,1", "1/1", "-b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.0"),
    "n3p02_2":    cfg("0,1", "1/1", "-b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.2"),
    "b32k_2":     cfg("0,1", "1/1", "-b 32768 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0"),
    "nomtp2":     cfg("0,1", "1/1", "-b 2048 -ub 2048"),
    "kf2":        cfg("0,1", "1/1", "-b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0",
                      binary="kf", env={"GGML_CUDA_DISABLE_MMVH": "1"}),
    # four cards
    "prod4":      cfg("0,1,2,3", "1/1/1/1", "-b 2048 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0"),
    "quick4":     cfg("0,1,2,3", "1/1/1/1", "-b 32768 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2"),
}

BIN = {"kmic": "/workspace/build-nccl/bin", "kf": f"{B}/kmicfrost/build/bin"}

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

def max_temp(gpus):
    out = sh(f"nvidia-smi -i {gpus} --query-gpu=temperature.gpu --format=csv,noheader").stdout.split()
    return max(int(x) for x in out) if out else 0

def cool(gpus, log):
    for _ in range(240):
        if max_temp(gpus) <= 48:
            return
        time.sleep(5)
    log(f"cool-timeout {max_temp(gpus)}")

def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:8099{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False

def prompts():
    text = open(WIKI, encoding="utf-8", errors="ignore").read()
    prose = text[20000:28000]   # ~2k tokens
    code = ("Write a complete, well-commented Python module that implements an LRU cache with TTL expiry, "
            "thread safety, statistics counters and a small CLI demo. Include type hints and unit tests.")
    return [("prose2k", f"Summarize the following article in detail, section by section:\n\n{prose}"),
            ("code", code)]

def main():
    out, rounds, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/log", "a")
    def log(m):
        logf.write(f"{time.strftime('%H:%M:%S')} {m}\n"); logf.flush()
    res = open(f"{out}/results.jsonl", "a")
    P = prompts()
    for r in range(1, rounds + 1):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            c = CONFIGS[name]
            cool(c["gpus"], log)
            envs = " ".join(f"-e {k}={v}" for k, v in c["env"].items())
            sh("docker rm -f srvab >/dev/null 2>&1")
            cmd = (f"docker run -d --name srvab --network host --gpus '\"device={c['gpus']}\"' {envs} "
                   f"-v {B}:{B} -v /home/user/p100-docker-test/build-nccl:/workspace/build-nccl:ro "
                   f"-v /home/user/models:/home/user/models:ro -v /data/models:/data/models:ro "
                   f"--entrypoint {BIN[c['bin']]}/llama-server p100-llamacpp-test:latest "
                   + c["args"].format(model=MODEL))
            log(f"r{r} start {name}")
            p = sh(cmd)
            if p.returncode != 0 or not wait_ready():
                log(f"r{r} {name} FAILED to start: {p.stderr.strip()[:200]}")
                sh(f"docker logs srvab > {out}/{name}.r{r}.fail.log 2>&1; docker rm -f srvab")
                continue
            try:
                post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello."}],
                                              "max_tokens": 32, "cache_prompt": False})
                for pname, ptxt in P:
                    t0 = time.time()
                    j = post("/v1/chat/completions", {"messages": [{"role": "user", "content": ptxt}],
                                                      "max_tokens": 512, "cache_prompt": False,
                                                      "chat_template_kwargs": {"enable_thinking": False}})
                    t = j.get("timings", {})
                    rec = {"round": r, "config": name, "prompt": pname, "wall_s": round(time.time() - t0, 2),
                           "prompt_n": t.get("prompt_n"), "pp_ts": t.get("prompt_per_second"),
                           "gen_n": t.get("predicted_n"), "tg_ts": t.get("predicted_per_second"),
                           "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                           "max_temp": max_temp(c["gpus"])}
                    res.write(json.dumps(rec) + "\n"); res.flush()
                    log(f"r{r} {name} {pname} pp={rec['pp_ts']} tg={rec['tg_ts']} temp={rec['max_temp']}")
                    if rec["max_temp"] > 80:
                        log("ABORT temp > 80"); sh("docker rm -f srvab"); return
            except Exception as e:
                log(f"r{r} {name} request error {e!r}")
            sh(f"docker logs srvab > {out}/{name}.r{r}.log 2>&1; docker rm -f srvab")
    log("SRV-DONE")

if __name__ == "__main__":
    main()
