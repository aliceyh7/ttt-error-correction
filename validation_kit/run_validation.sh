#!/usr/bin/env bash
# run_validation.sh -- ONE COMMAND, Mac.  Converts an official JAX/EasyLM TTT
# checkpoint to HF/PyTorch and runs the delta_fraction probe on a corpus.
#
#     ./run_validation.sh ~/ttt/linear-760m corpus.txt
#
# Argument 1 may be either the checkpoint FILE itself
# (~/ttt/linear-760m/streaming_train_state) or the DIRECTORY containing it.
# The converted model is cached next to it as <dir>/hf_converted/ and is
# reused on subsequent runs (delete that directory to force a re-convert).
#
# PREREQUISITES (once):
#     python3 -m pip install "torch" "transformers==4.44.2" safetensors msgpack numpy
#     # transformers >= 5 breaks TTTForCausalLM.post_init(); pin 4.4x.
#     # The released TTT models are Llama-2 tokenized; the default tokenizer is
#     # the un-gated mirror NousResearch/Llama-2-7b-hf.  Any Llama-2 tokenizer
#     # works:  TOKENIZER=meta-llama/Llama-2-7b-hf ./run_validation.sh ...
#     # No repo clones are needed: ttt.py is vendored in this kit.
#
# ENV OVERRIDES
#     MODEL=ttt_linear|ttt_mlp   default: inferred from the checkpoint
#     TOKENIZER=...              default NousResearch/Llama-2-7b-hf
#     MAXLEN=2048  WINDOWS=4     probe window length / how many windows
#     DEVICE=cpu|mps             default cpu (the probe is small matmuls; cpu wins)
#     OUTDIR=...                 where to put the converted model
#     SELFTEST=1                 run the synthetic conversion test first
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

usage() { echo "usage: $0 <checkpoint-file-or-dir> <corpus.txt>" >&2; exit 1; }
[ $# -ge 2 ] || usage

CKPT_IN="$1"
CORPUS="$2"
[ -e "$CKPT_IN" ] || { echo "no such path: $CKPT_IN" >&2; exit 1; }
[ -f "$CORPUS"  ] || { echo "no such corpus file: $CORPUS" >&2; exit 1; }

# ---- locate the msgpack checkpoint --------------------------------------
if [ -d "$CKPT_IN" ]; then
  CKPT=""
  for cand in streaming_train_state streaming_params; do
    if [ -f "$CKPT_IN/$cand" ]; then CKPT="$CKPT_IN/$cand"; break; fi
  done
  if [ -z "$CKPT" ]; then
    # e.g. streaming_train_state_29000 from a milestone save
    CKPT="$(find "$CKPT_IN" -maxdepth 2 -type f -name 'streaming_*' | head -1 || true)"
  fi
  [ -n "$CKPT" ] || { echo "no streaming_train_state/streaming_params under $CKPT_IN" >&2; exit 1; }
  BASE="$CKPT_IN"
else
  CKPT="$CKPT_IN"
  BASE="$(dirname "$CKPT")"
fi
OUTDIR="${OUTDIR:-$BASE/hf_converted}"

echo "=============================================================="
echo " checkpoint : $CKPT  ($(du -h "$CKPT" | cut -f1))"
echo " corpus     : $CORPUS  ($(wc -c < "$CORPUS") bytes)"
echo " converted  : $OUTDIR"
echo "=============================================================="

# ---- 0. optional self-test (synthetic; ~20 s, no downloads) -------------
if [ "${SELFTEST:-0}" = "1" ]; then
  echo; echo "### 0/3  self-test: synthetic checkpoint round trip"
  "$PY" "$HERE/test_conversion.py"
fi

# ---- 1. convert ---------------------------------------------------------
if [ -f "$OUTDIR/config.json" ]; then
  echo; echo "### 1/3  convert -- already converted, reusing $OUTDIR"
  echo "         (rm -rf '$OUTDIR' to force a re-convert)"
else
  echo; echo "### 1/3  convert JAX/EasyLM -> HF/PyTorch"
  "$PY" "$HERE/convert_jax_to_pt.py" "$CKPT" "$OUTDIR" --dtype float32 --verify
fi

# ---- 2. probe -----------------------------------------------------------
echo; echo "### 2/3  delta_fraction probe on real tokens"
"$PY" "$HERE/run_delta_probe.py" \
  --checkpoint "$OUTDIR" \
  --text "$CORPUS" \
  --tokenizer "${TOKENIZER:-NousResearch/Llama-2-7b-hf}" \
  --max-length "${MAXLEN:-2048}" \
  --windows "${WINDOWS:-4}" \
  --device "${DEVICE:-cpu}" \
  --eta-probe \
  --out "$OUTDIR/delta_fraction_probe.json"

# ---- 3. where to go next ------------------------------------------------
echo; echo "### 3/3  done"
echo "  table + JSON : $OUTDIR/delta_fraction_probe.json"
echo
echo "  The full five-arm ablation on this same converted checkpoint is:"
MODEL_GUESS="$("$PY" - "$OUTDIR" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
print("ttt_linear" if c["ttt_layer_type"] == "linear" else "ttt_mlp")
EOF
)"
echo "    python3 $HERE/run_ablation_ladder.py \\"
echo "        --checkpoint $OUTDIR --model ${MODEL:-$MODEL_GUESS} \\"
echo "        --tokenizer ${TOKENIZER:-NousResearch/Llama-2-7b-hf} \\"
echo "        --text $CORPUS --max-length 2048 --max-tokens 200000 \\"
echo "        --arms descent,hebbian,hebbian_nm,ascent,frozen --device ${DEVICE:-cpu} --tag official_760m"
