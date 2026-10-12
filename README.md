# Is Your Test-Time Learner Actually Learning? Error Correction in Trained TTT Layers

Code, harness, and raw results for the paper. Everything here runs against the **released,
unmodified** TTT-Linear / TTT-MLP checkpoints and the authors' reference `ttt.py`.

> Test-time-training layers update a fast-weight memory by gradient descent. Whether that
> update *revises* a stored binding (error correction, the delta rule) or only *adds* one is
> decided first by the inner loss and then by training. Dot-product losses (LaCT, ViTTT) have
> no error-correction term; the released MSE objective (TTT-Linear, TTT-MLP) has one, but gates
> it through a LayerNorm that is closed at initialization and only partly reopened by training.

## The three-step diagnostic (the paper's practical result)

Does a given KV-binding TTT layer error-correct? Check in order of increasing cost:

1. **Classify the inner loss from the code.** Dot-product → no error-correction term, and
   gradient ascent ≡ descent up to reparameterization (Prop. 1). Done.
2. **Read the LayerNorm gain from the checkpoint.** `CV(γ²) = 0` means the term is switched
   off exactly. Non-zero orders layers only approximately (ρ ≈ 0.3–0.5 vs. the live share).
   ```bash
   python validation_kit/convert_jax_to_pt.py <ckpt>/streaming_train_state converted/ --verify
   ```
3. **Probe the live gradient on real text.** This is the measurement.
   ```bash
   python validation_kit/run_delta_probe.py --checkpoint converted/ --text corpus.txt \
       --max-length 2048 --windows 4 --tokenizer NousResearch/Llama-2-7b-hf --eta-probe \
       --out probe.json
   python validation_kit/bootstrap_spearman.py probe.json
   ```
   Then, for behavior, the five-arm inference-time ladder (descent / Hebbian / Hebbian
   norm-matched / ascent / frozen):
   ```bash
   python validation_kit/run_ablation_ladder.py --checkpoint converted/ --model ttt_linear \
       --text corpus.txt --max-length 2048 --max-tokens 100000 \
       --arms descent,hebbian,hebbian_nm,ascent,frozen --device cuda --out-dir ladder/
   ```

## One-click reproduction

* **Modal** (what produced `results/modal/`): `modal run --detach modal_runs.py --stage all`
  from this directory. Downloads the checkpoints, converts them, runs the probes and ladders
  on an A10G, and writes everything to a persistent volume. About 6 GPU-hours total.
* **Colab**: open `colab_runs.ipynb`, Runtime → Run all on a GPU runtime. The kit is embedded.
* **Synthetic overwrite task** (CPU, minutes): `python scripts/overwrite_sweep.py --out results`
  then `python scripts/overwrite_fig.py`.

## Layout

```
validation_kit/      converter (JAX/EasyLM → HF PyTorch), live probe, five-arm ladder, arms,
                     bootstrap CI, synthetic-checkpoint equivalence test; vendors ttt.py verbatim
modal_runs.py        Modal job: prepare / probe / ladder / bootstrap stages on a volume
colab_runs.ipynb     same pipeline as a Colab notebook
scripts/             overwrite task sweep + figure scripts
results/modal/       raw JSON from the paper's runs (760M + 1.3B ladders, 1.3B + MLP-760M probes, CIs)
results/*.json       overwrite-task sweeps
figures/             Fig. 4 (ladder) and Fig. 5 (overwrite) as rendered
```

## What is verified, and how

* `validation_kit/test_conversion.py` builds byte-format-accurate synthetic EasyLM checkpoints,
  converts them, and compares the fp32 forward against a float64 NumPy transcription of the
  flax model, stage by stage, with seven deliberate mis-mappings as negative controls
  (worst relative error 6.6e-7).
* The `descent` arm re-installs the identical function object, so it is bitwise the
  unmodified model; the `frozen` arm provably leaves the fast weights untouched.
* The `hebbian_nm` arm rescales each token's gradient so every rank-one fast-weight write
  keeps the Frobenius norm of the unmodified write; the applied ratio is logged
  (`normmatch_ratio` in the ladder JSON).

## Checkpoints

Public, from `huggingface.co/Test-Time-Training`: `ttt-linear-760m-pile-8k`,
`ttt-linear-1.3b-pile-8k`, `ttt-mlp-760m-pile-8k` (and the five others for the parameter sweep).
Tokenizer: any Llama-2 tokenizer; the scripts default to the un-gated `NousResearch/Llama-2-7b-hf`.

## Citation

```bibtex
@article{hu2026ttt,
  title   = {Is Your Test-Time Learner Actually Learning? Error Correction in Trained TTT Layers},
  author  = {Hu, Yuhe and Hu, Wen Jia},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Our code: MIT (see `LICENSE`). `validation_kit/ttt.py` is the reference implementation from
`test-time-training/ttt-lm-pytorch`, vendored verbatim under its own MIT license
(`validation_kit/LICENSE.ttt-lm-pytorch`).
