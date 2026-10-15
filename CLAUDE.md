# NeuroSecure-Framework: project notes

Research code for the paper "Adversarial Vulnerability of Gradient-Trained Neural Networks:
Perturbation Limits in MNIST Image Classification" (Ethan Hahn, Tejas Chan, Laith Muflahi-Isley).
Python + PyTorch. See README.md for usage.

## Paper (Google Docs)
- Original draft: "Second Draft of Paper(PSI)", doc id `1gF19kYYRK5la35BjiQliHdOclL7cDOl3iYb8XRTCrmA` (left unchanged)
- Revised version: "Second Draft of Paper(PSI) - Revised", doc id `1Nr9oEqjXiSYkiEe69Q5D_ttwLP_p7NImge_vP6vwN0k`
  (same Drive folder). This is the current text; all numbers come from `mnist_project/results/summary.md`.
- No Google Docs editor connector is installed, so docs can only be created via Drive, not edited in place.

## Experimental setup (as actually run)
- Models: Simple MLP (784-512-10), Simple CNN (conv3x3 32ch, conv3x3 64ch, no padding, maxpool, 9216-128-10),
  Robust CNN (same + BatchNorm, dropout 0.25/0.5, random rotation ±15° on TRAINING images only; not adversarially trained).
- Training: Adam, lr 0.001, batch size 128. A fresh model per epoch budget 1..10 x 3 trials = 90 models.
- Attacks (`scripts/run_attacks.py`): model.eval(), l_inf, eps 0.05..0.5, clamp to [0,1], logs accuracy AND loss.
  FGSM: all 90 models, full 10k test set. PGD: 10-epoch models only, random start, 40 steps, step = eps/10,
  first 2,000 test images (`--pgd-n 2000`; script default is the full set).
- Paper reports 10-epoch models as mean ± std over 3 trials.

## Key results (10-epoch models)
- Clean acc: MLP 98.08±0.08, Simple CNN 98.90±0.09, Robust CNN 99.06±0.14 (loss 0.066 / 0.042 / 0.029).
- FGSM eps=0.1: 14.0 / 86.4 / 50.8 %. PGD eps=0.1: 2.9 / 66.6 / 9.7 %. PGD gives 0% for all models from eps=0.25.
- Ordering under attack: Simple CNN > Robust CNN > MLP at every eps. No crossover at 10 epochs; the earlier draft's
  crossover came from averaging over all epoch budgets (Robust CNN does lead at eps>=0.3 for 1-4 and 6 epochs).
- Simple CNN loss at eps=0.5: FGSM 10.7±2.2 vs PGD 79.7±4.0.
- Robust CNN collapses to predicting "8": 70% of test images at FGSM eps=0.2, 84% at eps=0.4. Large trial-to-trial spread
  (FGSM eps=0.05: 90.3 / 68.2 / 91.0 %). Verified this is NOT an eval-mode bug: old results reproduce exactly in eval mode.
- Overfitting is not supported (clean test loss flat or falling), so the paper presents overconfidence as a hypothesis.

## Status / open items
- All repo work is on branch `claude/festive-bardeen-bi4a34`, not yet merged to `main` (no PR opened yet).
- Model weights (*.pth, ~320 MB) are untracked. They still need uploading as `models.zip` to a GitHub Release
  (README points there); until then restore them with `git checkout 899720a -- mnist_project/models`.
- Revised Google Doc still needs: auto table of contents, page numbers, page breaks before Tables 2 and 3.
- Deliberately not done (would need history rewrite + force push): purging 320 MB from git history,
  rewording the old "Hello→Goodbye" commit message.
