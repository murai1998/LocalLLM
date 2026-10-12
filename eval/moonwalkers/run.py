"""Run the moonwalker prompt through each model preset via localllm-serve.

Usage (from the repo root, with the models downloaded): .venv/bin/python eval/moonwalkers/run.py
"""
import json, os, signal, subprocess, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "moonwalkers"
PROMPT = ("without using internet, prepare a list of all people who ever walked on the moon "
          "surface, provide 1-2 sentence background information on each")
RUNS = [("gemma-4-12b", "q5_k"), ("gemma-4-31b", ""), ("qwen3.8-27b", "")]
GW = "http://127.0.0.1:8090"

def wait_ready(proc, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("serve exited")
        try:
            if httpx.get(f"{GW}/health", timeout=2).json().get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError

for model, quant in RUNS:
    env = dict(os.environ, LOCALLLM_AUDIO_SIDECAR__MODE="off",
               LOCALLLM_LLAMA_SERVER__CONTEXT_SIZE="32768")
    cmd = [str(ROOT / ".venv/bin/localllm-serve"), "--model", model] + (["--quant", quant] if quant else [])
    log = open(OUT / f"{model}.serve.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc)
        for thinking in (False, True):
            body = {"messages": [{"role": "user", "content": PROMPT}],
                    "temperature": 0.7, "top_p": 0.95,
                    "max_tokens": 16000 if thinking else 4096,
                    "chat_template_kwargs": {"enable_thinking": thinking}}
            t = time.time()
            r = httpx.post(f"{GW}/v1/chat/completions", json=body, timeout=3600).json()
            msg = r["choices"][0]["message"]
            rec = {"model": model, "quant": quant or "default", "thinking": thinking,
                   "elapsed_sec": round(time.time() - t, 1),
                   "finish_reason": r["choices"][0].get("finish_reason"),
                   "usage": r.get("usage"), "timings": r.get("timings"),
                   "content": msg.get("content"), "reasoning": msg.get("reasoning_content")}
            tag = f"{model}.{'think' if thinking else 'nothink'}"
            (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            (OUT / f"{tag}.md").write_text(rec["content"] or "")
            print(f"DONE {tag} {rec['elapsed_sec']}s finish={rec['finish_reason']}", flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["pkill", "-f", "llama-server"])
        time.sleep(5)
print("ALL DONE", flush=True)
