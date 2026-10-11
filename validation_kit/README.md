# TTT validation kit — official JAX checkpoint → PyTorch → delta_fraction probe

Everything needed to run the `delta_fraction` validation on the **official**
Test-Time-Training checkpoints, on a Mac, with no repo clones.

You need exactly:

* this directory,
* your checkpoint (the EasyLM `streaming_train_state` msgpack file from
  `huggingface.co/Test-Time-Training/ttt-linear-760m-pile-8k` etc.),
* `python3` with `torch`, `transformers==4.4x`, `safetensors`, `msgpack`, `numpy`.

```bash
python3 -m pip install "torch" "transformers==4.44.2" safetensors msgpack numpy
```

> `transformers >= 5` breaks `TTTForCausalLM.post_init()`. Pin 4.4x.

---

## One command

```bash
./run_validation.sh ~/ttt/linear-760m corpus.txt
```

Argument 1 is the directory holding `streaming_train_state` (or the file
itself); argument 2 is any plain-text file. The script

1. converts the checkpoint to an HF/PyTorch directory at
   `~/ttt/linear-760m/hf_converted/` (cached — delete it to re-convert),
   verifying it with a real forward pass;
2. runs `delta_fraction_probe` over 4 windows of 2048 real tokens and prints
   the per-layer table of
   `delta_fraction = ||g_descent − g_hebbian|| / ||g_descent||`
   next to `CV(gamma^2)`, `gamma_rms`, `beta_rms` and `eta·||k||²`;
3. writes `hf_converted/delta_fraction_probe.json` and prints the command for
   the full four-arm ablation on the same converted model.

Useful environment overrides: `TOKENIZER`, `MAXLEN`, `WINDOWS`, `DEVICE`
(`cpu` or `mps`), `OUTDIR`, `SELFTEST=1`.

### Tokenizer

The released TTT models are **Llama-2 tokenized** (`vocab_size = 32000`), so
the default is `meta-llama/Llama-2-7b-hf`, a *gated* repo — run
`huggingface-cli login` after accepting the licence, or point `TOKENIZER=` at
any Llama-2 tokenizer mirror you already have. `run_delta_probe.py` fails
loudly if the tokenizer's ids overflow the model vocabulary.

---

## Self-test (no downloads, ~30 s)

```bash
python3 test_conversion.py
```

Builds synthetic checkpoints in the exact EasyLM byte format (bf16 payloads
included), converts them, loads them with the authors' unmodified `ttt.py`,
and compares the fp32 forward pass — stage by stage — against
`jax_ref_numpy.py`, a float64 NumPy transcription of the flax model run on the
*unconverted* flax-layout arrays. Four architecture variants are covered:
TTT-Linear and TTT-MLP with the released Mamba-style backbone
(`share_qk` + gate + `pre_conv`), the `*_base` backbone (separate `wk`, no
conv, no gate), and an untied `lm_head` written as `streaming_params`.

It then runs **negative controls**: seven deliberately wrong versions of the
layout/name map (dropped transposes, a wrong conv permutation, time-flipped
conv taps, a transposed `W1`, swapped `conv_q`/`conv_k`, swapped
`gate_proj`/`up_proj`). All seven must be detected, so the test cannot pass
vacuously. Skip with `--no-mutations`.

Observed agreement: worst relative error **6.6e-7** across all stages and
variants — pure fp32 round-off.

---

## Troubleshooting a real checkpoint

The converter has never been run against a real released file (this kit was
built in an environment with no huggingface.co access), so the first thing to
do if anything looks wrong is dump what is actually in the file:

```bash
python3 convert_jax_to_pt.py ~/ttt/linear-760m/streaming_train_state --list-keys | head -60
```

Every record is listed with its shape, dtype and the torch name it maps to
(`-` = not mapped, e.g. `step` and the Adam moments). Anything under
`params/params/model/…` that shows `-` is a parameter this converter does not
know about, and the run will stop with a hard error rather than silently
leaving a randomly-initialised tensor in the model.

Memory: shards are written and freed as they fill, so peak RSS ≈
`--max-shard-size` (default 1 GB) + the largest single tensor + ~100 MB. For
the 760M model that is roughly 1.3 GB; drop to `--max-shard-size 0.5` on a
tight machine. The fp32 output is ~3.0 GB on disk for the 760M model.

---

## Pieces

| file | what it is |
|---|---|
| `run_validation.sh` | the one-command driver |
| `convert_jax_to_pt.py` | EasyLM/flax msgpack → HF safetensors + `config.json`; streams the file and writes sharded output, so peak RSS ≈ one shard, not the whole model. `--verify` loads and runs the result |
| `run_delta_probe.py` | loads a converted model, runs exp3's `delta_fraction_probe`, prints/saves the per-layer table |
| `run_ablation_ladder.py` | exp3's full four-arm ablation (vendored) |
| `ttt_arms.py` | exp3's arm/probe implementation (vendored; one edit, see `MANIFEST.txt`) |
| `ttt.py` | `ttt-lm-pytorch/ttt.py`, **verbatim** — the model definition |
| `jax_ref_numpy.py` | float64 NumPy transcription of the flax forward, reference side of the equivalence test |
| `make_synthetic_jax_ckpt.py` | writes byte-format-accurate fake EasyLM checkpoints |
| `test_conversion.py` | the equivalence test |

---

## The conversion, in one table

flax kernels are `[in, out]`, torch `nn.Linear` weights are `[out, in]`;
flax `nn.Conv` kernels are `[width, in/groups, C]`, torch `nn.Conv1d` weights
are `[C, in/groups, width]`. Everything else is a rename.

| flax (`params/params/…`) | torch | transform |
|---|---|---|
| `model/wte/embedding` | `model.embed_tokens.weight` | — |
| `model/ln_f/kernel` | `model.norm.weight` | — |
| `lm_head/kernel` *(absent when tied)* | `lm_head.weight` | transpose |
| `model/h/N/seq_norm/kernel` | `model.layers.N.seq_norm.weight` | — |
| `model/h/N/ffn_norm/kernel` | `model.layers.N.ffn_norm.weight` | — |
| `model/h/N/feed_forward/w1/kernel` | `…mlp.gate_proj.weight` | transpose |
| `model/h/N/feed_forward/w3/kernel` | `…mlp.up_proj.weight` | transpose |
| `model/h/N/feed_forward/w2/kernel` | `…mlp.down_proj.weight` | transpose |
| `model/h/N/conv/conv_norm/kernel` | `…conv.norm.weight` | — |
| `model/h/N/conv/conv1/kernel` | `…conv.conv.weight` | `(2,1,0)` |
| `model/h/N/conv/conv1/bias` | `…conv.conv.bias` | — |
| `…/seq_modeling_block/wq/kernel` | `…seq_modeling_block.q_proj.weight` | transpose |
| `…/seq_modeling_block/wk/kernel` *(only if `share_qk=False`)* | `…k_proj.weight` | transpose |
| `…/seq_modeling_block/wv/kernel` | `…v_proj.weight` | transpose |
| `…/seq_modeling_block/wo/kernel` | `…o_proj.weight` | transpose |
| `…/seq_modeling_block/wg/kernel` | `…g_proj.weight` | transpose |
| `…/seq_modeling_block/conv_q/{kernel,bias}` | `…conv_q.{weight,bias}` | `(2,1,0)` / — |
| `…/seq_modeling_block/conv_k/{kernel,bias}` | `…conv_k.{weight,bias}` | `(2,1,0)` / — |
| `…/seq_modeling_block/ttt_dense_0` | `…W1` | — |
| `…/seq_modeling_block/ttt_bias_0` | `…b1` | — |
| `…/seq_modeling_block/ttt_dense_1` *(TTT-MLP)* | `…W2` | — |
| `…/seq_modeling_block/ttt_bias_1` *(TTT-MLP)* | `…b2` | — |
| `…/seq_modeling_block/ttt_norm/scale` | `…ttt_norm_weight` | — |
| `…/seq_modeling_block/ttt_norm/bias` | `…ttt_norm_bias` | — |
| `…/seq_modeling_block/post_norm/{scale,bias}` | `…post_norm.{weight,bias}` | — |
| `…/seq_modeling_block/learnable_ttt_lr/kernel` | `…learnable_ttt_lr_weight` | `(0,2,1)` |
| `…/seq_modeling_block/learnable_ttt_lr/bias` | `…learnable_ttt_lr_bias` | — |
| `…/seq_modeling_block/learnable_token_idx` | `…learnable_token_idx` | — |

Four things are easy to get wrong and are all covered by `test_conversion.py`:

* **`share_qk`.** The released TTT models use `TTTLinear`/`TTTMLP` (not the
  `*_base` classes), whose `setup_qkvo` has **no `wk`**: it computes
  `xqk = wq(x)` and then two separate causal depthwise convolutions
  `XQ = conv_q(xqk)`, `XK = conv_k(xqk)`. `ttt-lm-pytorch` supports exactly
  this under `share_qk=True`, with the same names (`q_proj`, `conv_q`,
  `conv_k`) and the same semantics.
* **Conv layout.** flax `[width, 1, C]` → torch `[C, 1, width]`, i.e. a
  `(2,1,0)` permutation, *not* a reversal of the taps. flax `padding="CAUSAL"`
  and torch `padding=kernel-1` + truncate agree tap for tap.
* **The learnable-η gate.** flax stores a head-stacked `Dense(width→1)` kernel,
  `[nh, width, 1]`; torch wants `[nh, 1, width]`. Its `bias` is `[nh, 1]` in
  both and must **not** be transposed.
* **`W1` / `b1`.** Already head-major in flax (`[nh, f, f]`, `[nh, 1, f]`) and
  copied through unchanged. No reshape of a flat buffer is involved — the
  760M file's `ttt_dense_0` is 16·96·96·2 B = 294 912 B of payload.

**RoPE needs no fix-up.** flax rotates interleaved pairs `(x₂ⱼ, x₂ⱼ₊₁)`;
`ttt.py` uses the `rotate_half` convention but wraps it in
`permute_qk(...)` / `undo_permute_qk(...)`, which is exactly the change of
basis between the two. Do **not** permute anything in the converter.

---

## Config fields inferred from the checkpoint

`vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`,
`num_attention_heads`, `mini_batch_size` (length of `learnable_token_idx`),
`conv_kernel`, `ttt_layer_type` (`mlp` iff `ttt_dense_1` present), `share_qk`
(iff `conv_q` present), `use_gate` (iff `wg` present), `pre_conv` (iff
`conv/conv1` present), `tie_word_embeddings` (iff no `lm_head/kernel`).

**Not** stored in the checkpoint and therefore taken from the JAX configs in
`ttt-lm-jax/ttt/models/model.py` (`*-TTT` entries), overridable on the command
line: `--rope-theta 10000.0`, `--ttt-base-lr 1.0`, `--rms-norm-eps 1e-6`.
Both defaults are correct for every released TTT model. `ttt_base_lr` scales η
directly, so if a checkpoint were trained with a different value the probe's
`eta*key^2` column would be wrong (`delta_fraction` and `CV(gamma^2)` would
not be).

Finally, the converter asserts that the resulting key set is *exactly* the
`TTTForCausalLM` state dict for the inferred config — a missing or extra
tensor is a hard error, never a silently randomly-initialised weight. With
`tie_word_embeddings`, `--verify` additionally checks that `transformers`
really tied `lm_head` to the embedding (shared storage), because a silent
failure there would leave a random output head.

---

## What has *not* been verified

* No real released checkpoint was ever opened. The byte format, the key paths
  and the `params/params/` prefix come from `ttt-lm-jax/ttt/infra/checkpoint.py`
  + `flax.serialization`, and from the parameter list observed on the real 760M
  file; the synthetic fixtures reproduce that format exactly, but a surprise in
  the real file (an extra parameter, a different prefix, a milestone-suffixed
  filename) would surface as a hard error, not as wrong numbers. Run
  `--list-keys` first.
* `rope_theta`, `ttt_base_lr` and `rms_norm_eps` are not in the checkpoint and
  are taken from the JAX `*-TTT` configs. They are correct for every released
  model but are assumptions, not measurements.
* Equivalence was checked against a NumPy transcription of the flax code, not
  against JAX/flax executing. The transcription is line-referenced to the
  source in `jax_ref_numpy.py`; a shared misreading of the flax semantics by
  both the converter and the transcription would not be caught. The one place
  this could bite is the LayerNorm VJP inside the TTT inner loop, which is
  written out analytically here rather than obtained from `jax.vjp`.
