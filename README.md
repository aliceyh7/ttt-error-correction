# Error Correction in Trained TTT Layers

Code and saved results for the revised paper by Yuhe Hu.

The analysis distinguishes an explicit prediction-derived gradient from its behavioral effect. The released LayerNorm objective retains state dependence after prediction removal. The historical arm names `hebbian` and `hebbian_nm` remain in the code and result files for compatibility. They mean prediction removed and prediction removed with local norm matching in the revised paper.

## Reproduce the revised analysis

```sh
python -m pip install -r requirements-analysis.txt
python scripts/recompute_and_check.py
python validation_kit/test_conversion.py
python scripts/make_revision_figures.py
```

The analysis writes summaries and numerical checks to `results/revision/`. The figure script creates the three revised figures in `figures/`. The two older figure entry points call this script as well.

The analysis recomputes the gradient ratios, perplexity effects, and paired layer-bootstrap intervals. Bootstrap ranks now average ties, including observations duplicated by resampling. It verifies the LayerNorm decomposition against automatic differentiation, the finite-epsilon residual, local norm matching, a gain-orientation counterexample, a nonlinear sign-symmetry example, baseline identity, and frozen-state behavior. A closed-form calculation independently reproduces the 28 stored two-write simulation cells.

The conversion test checks synthetic checkpoint fixtures against a NumPy reference and includes seven deliberate mapping errors. These checks do not rerun the original full-checkpoint GPU experiments.

## Results and interpretation

The saved language-model ablations score 100303 paired token positions from War and Peace on each linear checkpoint. Prediction removal with local norm matching increases perplexity by about 2.8 to 2.9 percent. Freezing fast-weight updates increases it by factors of about 4.2 to 4.5. The evaluation is one text that may overlap pretraining, not a held-out multi-document or explicit semantic revision benchmark.

The mean measured norm ratios are 0.265 for TTT-Linear-760M, 0.251 for TTT-Linear-1.3B, and 0.318 for TTT-MLP-760M. The saved 760M probe covers 128 chunks per layer. The standalone 1.3B and MLP probes each cover 512.

The default `drop_pred` intervention deletes the full affine normalized prediction, including the bias. `drop_w` deletes only the gain-scaled normalized prediction. The retained target gradient remains state dependent through the normalization Jacobian. A uniform gain cancels the gain-scaled component only in the zero-epsilon limit. Gain variation is not a calibrated gate opening, and average norm ratios do not establish component orthogonality.

Norm matching preserves a token/head gradient norm at the current modified state. It does not preserve the summed chunk-update norm or the separate baseline trajectory. The paper reports descriptive paired effects without interpreting token-based standard errors as independent-document uncertainty.

The overwrite figure uses the saved two-write simulation. Its repeated-write panel is an exact identical-key calculation with orthogonal values. It is not a measurement of trained-language-model overwrite speed. Older repeated-noisy-write results are retained as historical data but are not used by the revised analysis.

## Full model runs

```sh
modal run --detach modal_runs.py --stage all
```

This downloads and converts public checkpoints and runs probes and ablations on Modal GPUs. It uses paid compute and can take hours. Completed stages are cached. The Colab notebook contains the same validation kit, including the corrected bootstrap. The original saved run environment uses PyTorch 2.4.1 and Transformers 4.44.2. The revision's local checks used PyTorch 2.14.1 and Transformers 4.44.2.

## Repository contents

- `validation_kit/` contains the reference model, converter, probes, interventions, and bootstrap.
- `scripts/` contains the revised analysis and figure generation, plus the original overwrite sweep.
- `results/modal/` contains the original saved model outputs.
- `results/revision/` contains recomputed summaries, checks, and synthetic notebook records.
- `figures/` contains revised vector PDFs and PNG previews.
- `modal_runs.py` and `colab_runs.ipynb` run the full checkpoint pipeline.

The reference model in `validation_kit/ttt.py` is unchanged. Its license is in `validation_kit/LICENSE.ttt-lm-pytorch`. Other experiment code is covered by the repository MIT license.
