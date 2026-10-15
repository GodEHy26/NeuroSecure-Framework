# NeuroSecure-Framework

Code and results for the paper **"Adversarial Vulnerability of Gradient-Trained Neural Networks:
Perturbation Limits in MNIST Image Classification"** by Ethan Hahn, Tejas Chan and Laith Muflahi-Isley.

We train three image classifiers on MNIST and measure how their accuracy and loss degrade under two
white-box, l∞-bounded adversarial attacks: the Fast Gradient Sign Method (FGSM) and Projected Gradient
Descent (PGD). Everything is written in Python with PyTorch.

## The models

| Model | Layers |
|---|---|
| Simple MLP | 784 → 512 (ReLU) → 10 |
| Simple CNN | Conv 3×3, 32 ch → ReLU → Conv 3×3, 64 ch → ReLU → MaxPool 2×2 → 9216 → 128 (ReLU) → 10. Stride 1, no padding, so 28 → 26 → 24 → 12 and 64·12·12 = 9216 |
| Robust CNN | Same convolutional and dense layers as the Simple CNN, plus batch norm after each convolution, dropout 0.25 after pooling and 0.5 after the dense layer, and random rotation (±15°) of the **training** images. "Robust" refers to these standard regularizers. The model is **not** adversarially trained |

## Experimental protocol

* **Training:** Adam, learning rate 0.001, batch size 128, cross-entropy loss. For each epoch budget
  E = 1…10 and each of 3 trials, a fresh model is trained from scratch for E epochs, giving 90 models
  (3 architectures × 10 budgets × 3 trials). Each is evaluated on the clean 10,000-image test set.
* **Attacks:** l∞ threat model, ε ∈ {0.05, 0.10, …, 0.50}, pixel values in [0, 1]. Models are attacked in
  `eval()` mode (batch-norm running statistics, dropout off). Every adversarial image is clamped to [0, 1].
  * **FGSM:** x* = clip(x + ε·sign(∇ₓL), 0, 1). Run on all 90 models and the full test set.
  * **PGD:** random uniform start inside the ε-ball, 40 steps of size α = ε/10, and after every step a
    projection back onto the ε-ball followed by clamping to [0, 1]. Run on the 10-epoch models. The committed
    results use the first 2,000 test images (the script's default is the full test set).
* Accuracy **and** mean cross-entropy loss are logged for every (model, attack, ε). Results are reported as
  mean ± standard deviation over the 3 trials.

## Headline results

10-epoch models, mean ± std over 3 trials. FGSM uses all 10,000 test images; PGD uses the first 2,000.

| Model | Clean acc. (%) | FGSM ε=0.1 (%) | PGD ε=0.1 (%) | FGSM ε=0.3 (%) | PGD ε=0.3 (%) |
|---|---|---|---|---|---|
| Simple MLP | 98.08 ± 0.08 | 14.0 ± 0.3 | 2.9 ± 0.3 | 0.2 ± 0.1 | 0.0 ± 0.0 |
| Simple CNN | 98.90 ± 0.09 | 86.4 ± 1.4 | 66.6 ± 10.3 | 24.8 ± 4.7 | 0.0 ± 0.0 |
| Robust CNN | 99.06 ± 0.14 | 50.8 ± 17.3 | 9.7 ± 1.9 | 14.7 ± 4.9 | 0.0 ± 0.0 |

* Convolutional structure helps: the Simple CNN is the most robust model at every ε.
* The regularizers do not: the Robust CNN is less robust than the Simple CNN, varies strongly between
  trials, and under attack predicts "8" for most inputs (70% of all test images at FGSM ε=0.2).
* PGD is much stronger than FGSM: every model is at 0% from ε=0.25, where FGSM still leaves the CNNs
  at roughly 10–35%.

See [`mnist_project/results/summary.md`](mnist_project/results/summary.md) for the full tables, and
[`mnist_project/visuals/figures/`](mnist_project/visuals/figures) for the figures.

## Repository layout

```
mnist_project/
  scripts/
    common.py              shared paths, model definitions, data loading, evaluation
    train_models.py        trains the 90 models and logs clean accuracy / loss
    evaluate_clean.py      re-checks clean accuracy / loss of the saved models
    run_attacks.py         FGSM + PGD attacks -> results CSV, confusion matrices, sample images
    summarize_results.py   mean ± std tables (results/summary*.{csv,md}) and paper figures
    generate_3d_graph.py   accuracy surface over (ε, epochs); interpolated, with measured points marked
    generate_gif.py        animated confusion matrix as ε grows
    view_confusion_matrix.py
  results/
    training_results.csv           per model: training time, clean accuracy, clean loss
    adversarial_attack_results.csv per model × attack × ε: accuracy and loss under attack
    summary_*.csv, summary.md      aggregated tables
  visuals/
    figures/  3d_graphs/  gifs/  samples/  confusion_matrices/{clean,fgsm_<ε>,pgd_<ε>}/
  models/     trained weights (*.pth, not tracked in git, see below)
data/         MNIST (downloaded automatically, not tracked in git)
```

## Getting started

```bash
git clone https://github.com/godehy26/neurosecure-framework.git
cd neurosecure-framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All scripts resolve paths relative to the repository, so they can be run from any directory.

### Trained models

The 90 trained models (~320 MB) are not stored in the git tree. Either:

* download `models.zip` from the repository's GitHub **Releases** page and unzip it into `mnist_project/models/`, or
* restore them from the commit that originally added them:
  `git checkout 899720a -- mnist_project/models && git reset -q mnist_project/models`, or
* retrain them with `python mnist_project/scripts/train_models.py`. It refuses to replace existing model files or
  `results/training_results.csv` unless you pass `--overwrite` (or `--log <path>` for a separate log).

### Reproducing the results

```bash
python mnist_project/scripts/evaluate_clean.py            # checks results/training_results.csv
python mnist_project/scripts/run_attacks.py --pgd-n 2000  # as committed; omit --pgd-n for the full test set
python mnist_project/scripts/summarize_results.py
python mnist_project/scripts/generate_3d_graph.py
python mnist_project/scripts/generate_gif.py --model robust_cnn
```

On a 4-core CPU, FGSM on all 90 models takes about 40 minutes, and PGD on the 10-epoch models
with 2,000 images takes about 1.5 hours. A GPU is used automatically if one is available.

## The team

* **Ethan Hahn**
* **Tejas Chan**
* **Laith Muflahi-Isley**

## License

MIT, see [LICENSE](LICENSE).
