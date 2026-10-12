"""TTT paper revision experiments on Modal.

Everything persists on a Modal Volume ("ttt-revision"): downloaded checkpoints,
converted PyTorch models, and results. Re-running skips finished stages.

    modal run --detach modal_runs.py --stage e1          # 5-arm ladder, 760M      (~2 h A10G)
    modal run --detach modal_runs.py --stage e3          # bootstrap CIs            (seconds, CPU)
    modal run --detach modal_runs.py --stage e6-probe    # probes 1.3B + MLP-760M   (~40 min)
    modal run --detach modal_runs.py --stage e6-ladder   # 5-arm ladder, 1.3B       (~3-4 h)
    modal run --detach modal_runs.py --stage all
    modal run modal_runs.py --stage status               # what's done
    modal volume get ttt-revision results ./results      # pull results locally

GPU: A10G (edit the gpu= arguments below for A100, ~2x faster on the ladders).
Run from the directory that contains validation_kit/ (the patched v2 kit).
"""
import os
import modal

APP = "ttt-revision"
app = modal.App(APP)
vol = modal.Volume.from_name(APP, create_if_missing=True)
V = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy<2", "transformers==4.44.2", "safetensors",
                 "msgpack", "huggingface_hub", "matplotlib", "tqdm", "scipy")
    .add_local_dir("validation_kit", remote_path="/root/validation_kit")
)

CKPTS = {
    "linear-760m": "Test-Time-Training/ttt-linear-760m-pile-8k",
    "linear-1.3b": "Test-Time-Training/ttt-linear-1.3b-pile-8k",
    "mlp-760m":    "Test-Time-Training/ttt-mlp-760m-pile-8k",
}
MODEL_KIND = {"linear-760m": "ttt_linear", "linear-1.3b": "ttt_linear", "mlp-760m": "ttt_mlp"}
TOK = "NousResearch/Llama-2-7b-hf"
CORPUS_URL = "https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt"
ARMS = "descent,hebbian,hebbian_nm,ascent,frozen"


def _run(cmd, cwd="/root/validation_kit"):
    import subprocess, sys
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")


def _corpus():
    import urllib.request
    p = f"{V}/corpus.txt"
    if not os.path.exists(p):
        urllib.request.urlretrieve(CORPUS_URL, p)
        vol.commit()
    return p


@app.function(image=image, volumes={V: vol}, timeout=2 * 3600, cpu=4, memory=16384)
def prepare(name: str):
    """Download + convert one checkpoint (CPU job; cached on the volume)."""
    from huggingface_hub import snapshot_download
    conv = f"{V}/converted/{name}"
    if os.path.exists(f"{conv}/config.json"):
        print("converted cached:", name); return name
    ck = f"{V}/ckpts/{name}"
    if not os.path.exists(f"{ck}/streaming_train_state"):
        snapshot_download(repo_id=CKPTS[name], local_dir=ck)
        vol.commit()
    _run(["convert_jax_to_pt.py", f"{ck}/streaming_train_state", conv, "--verify"])
    vol.commit()
    return name


@app.function(image=image, volumes={V: vol}, gpu="A10G", timeout=6 * 3600)
def ladder(name: str):
    import json
    out_dir = f"{V}/results/ladder_{name}"
    tag = name.replace("-", "").replace(".", "")
    out = f"{out_dir}/ablation_results_{tag}.json"
    if os.path.exists(out):
        print("cached:", out)
    else:
        corpus = _corpus()
        _run(["run_ablation_ladder.py",
              "--checkpoint", f"{V}/converted/{name}", "--model", MODEL_KIND[name],
              "--tokenizer", TOK, "--text", corpus,
              "--max-length", "2048", "--max-tokens", "100000",
              "--arms", ARMS, "--hebbian-mode", "drop_pred", "--normmatch-level", "token",
              "--device", "cuda", "--tag", tag, "--out-dir", out_dir])
        vol.commit()
    d = json.load(open(out))
    return {a: round(v["ppl"], 4) for a, v in d["per_arm"].items()} | \
           {"normmatch_ratio_mean": d.get("normmatch_ratio", {}).get("mean")}


@app.function(image=image, volumes={V: vol}, gpu="A10G", timeout=2 * 3600)
def probe(name: str):
    import json
    out = f"{V}/results/probe_{name}.json"
    if os.path.exists(out):
        print("cached:", out)
    else:
        corpus = _corpus()
        _run(["run_delta_probe.py",
              "--checkpoint", f"{V}/converted/{name}", "--text", corpus,
              "--max-length", "2048", "--windows", "4", "--device", "cuda",
              "--tokenizer", TOK, "--eta-probe", "--out", out])
        vol.commit()
    return json.load(open(out)).get("summary")


@app.function(image=image, volumes={V: vol}, timeout=600)
def bootstrap():
    import glob, json
    res = {}
    for src in sorted(glob.glob(f"{V}/results/ladder_*/ablation_results_*.json") +
                      glob.glob(f"{V}/results/probe_*.json")):
        out = f"{V}/results/spearman_{os.path.basename(src).replace('.json', '')}.json"
        _run(["bootstrap_spearman.py", src, "--out", out])
        res[os.path.basename(src)] = json.load(open(out))
    vol.commit()
    return res


@app.function(image=image, volumes={V: vol}, timeout=300)
def status():
    import glob
    return {"ckpts": sorted(os.path.basename(p) for p in glob.glob(f"{V}/ckpts/*")),
            "converted": sorted(os.path.basename(p) for p in glob.glob(f"{V}/converted/*")),
            "results": sorted(p.replace(V + "/", "") for p in glob.glob(f"{V}/results/**", recursive=True)
                              if os.path.isfile(p))}


@app.local_entrypoint()
def main(stage: str = "all"):
    if stage == "status":
        print(status.remote()); return
    if stage in ("e1", "all"):
        prepare.remote("linear-760m")
        print("E1 ladder 760M:", ladder.remote("linear-760m"))
    if stage in ("e3", "e1", "all"):
        print("E3 bootstrap:", bootstrap.remote())
    if stage in ("e6-probe", "all"):
        for n in ("linear-1.3b", "mlp-760m"):
            prepare.remote(n)
        for n, s in zip(("linear-1.3b", "mlp-760m"), probe.map(["linear-1.3b", "mlp-760m"])):
            print(f"E6 probe {n}:", s)
        print("E6 bootstrap:", bootstrap.remote())
    if stage in ("e6-ladder", "all"):
        prepare.remote("linear-1.3b")
        print("E6 ladder 1.3B:", ladder.remote("linear-1.3b"))
    print("done; pull with: modal volume get ttt-revision results ./results")
