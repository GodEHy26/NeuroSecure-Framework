"""Run FGSM and PGD attacks against the saved models.

Writes:
  results/adversarial_attack_results.csv   one row per (model, epochs, trial, attack, epsilon)
  visuals/confusion_matrices/<attack>_<eps>/<model>_epXX_tY_cm.npy  (plus clean/)
  visuals/samples/<model>/digit_<d>/sample_<k>.png and sample_<k>_data.json

Both attacks:
  * run with the model in eval mode (batch norm uses running statistics, dropout is off)
  * use the l_inf threat model and clamp every adversarial image to the valid pixel range [0, 1]
  * log both accuracy and mean cross-entropy loss on the adversarial images

PGD (Madry et al., 2018) starts from a uniformly random point in the eps-ball, takes
--pgd-steps signed-gradient steps of size alpha = eps * --pgd-step-frac, and after every
step projects back onto the eps-ball around the clean image and clamps to [0, 1].
With the defaults (40 steps, alpha = eps/10) the attack can travel 4*eps in total,
so it can reach the boundary of the eps-ball at every epsilon.

Usage (from anywhere):
  python mnist_project/scripts/run_attacks.py                     # everything, full test set
  python mnist_project/scripts/run_attacks.py --attacks fgsm      # FGSM only
  python mnist_project/scripts/run_attacks.py --pgd-n 2000        # PGD on first 2000 test images
"""
import argparse
import csv
import json
import os
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import (ATTACK_CSV, CM_DIR, EPOCHS, EPSILONS, MODEL_LABELS, SAMPLES_DIR, TRIALS,
                    evaluate, get_device, load_model, test_tensors)

FIELDS = ["Model_Type", "Epochs", "Trial", "Attack", "Epsilon", "N_Images", "Adv_Accuracy", "Adv_Loss"]


def input_gradient(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad, = torch.autograd.grad(loss, x)
    return grad


def fgsm(model, x, y, eps, batch_size=1000, grad_sign=None):
    """x* = clamp(x + eps * sign(grad_x L), 0, 1). grad_sign may be precomputed (it does not depend on eps)."""
    if grad_sign is None:
        grad_sign = torch.cat([input_gradient(model, x[i:i + batch_size], y[i:i + batch_size]).sign()
                               for i in range(0, len(x), batch_size)])
    return (x + eps * grad_sign).clamp(0, 1)


def pgd(model, x, y, eps, steps=40, step_frac=0.1, random_start=True, batch_size=1000, seed=0):
    alpha = eps * step_frac
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        if random_start:
            noise = (torch.rand(xb.shape, generator=gen) * 2 - 1).to(xb.device) * eps
            x_adv = (xb + noise).clamp(0, 1)
        else:
            x_adv = xb.clone()
        for _ in range(steps):
            x_adv = x_adv + alpha * input_gradient(model, x_adv, yb).sign()
            x_adv = torch.min(torch.max(x_adv, xb - eps), xb + eps).clamp(0, 1)   # project, then clamp
        out.append(x_adv.detach())
    return torch.cat(out)


def confusion_matrix(y, preds):
    cm = torch.zeros(10, 10, dtype=torch.long)
    idx = (y.cpu() * 10 + preds.cpu())
    cm.view(-1).index_add_(0, idx, torch.ones_like(idx))
    return cm.numpy()


def save_cm(folder, label, ep, trial, cm):
    path = os.path.join(CM_DIR, folder)
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, f"{label}_ep{ep:02d}_t{trial}_cm.npy"), cm)


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["Model_Type"], int(r["Epochs"]), int(r["Trial"]), r["Attack"], float(r["Epsilon"]))
                for r in csv.DictReader(f)}


def save_samples(model, label, x, y, args, per_digit=3):
    """For the first `per_digit` test images of each digit, store adversarial images and class probabilities."""
    for digit in range(10):
        idxs = (y == digit).nonzero().flatten()[:per_digit]
        folder = os.path.join(SAMPLES_DIR, label, f"digit_{digit}")
        os.makedirs(folder, exist_ok=True)
        for k, idx in enumerate(idxs.tolist(), start=1):
            xi, yi = x[idx:idx + 1], y[idx:idx + 1]
            data = {"test_index": idx, "true_label": digit, "model": f"{label}_ep{args.sample_epochs:02d}_t1",
                    "probs": {}}
            rows = {}
            for attack in args.attacks:
                imgs, titles = [], []
                for eps in [0.0] + EPSILONS:
                    if eps == 0.0:
                        xa = xi
                    elif attack == "fgsm":
                        xa = fgsm(model, xi, yi, eps)
                    else:
                        xa = pgd(model, xi, yi, eps, args.pgd_steps, args.pgd_step_frac, seed=idx)
                    with torch.no_grad():
                        p = F.softmax(model(xa), 1)[0]
                    key = "eps_0.0" if eps == 0.0 else f"{attack}_eps_{eps:g}"
                    data["probs"][key] = [round(v, 6) for v in p.tolist()]
                    imgs.append(xa[0, 0].cpu().numpy())
                    titles.append(f"ε={eps:g}\npred {p.argmax().item()} ({p.max().item():.2f})")
                rows[attack] = (imgs, titles)
            fig, axes = plt.subplots(len(rows), len(EPSILONS) + 1, figsize=(2 * (len(EPSILONS) + 1), 2.9 * len(rows)),
                                     squeeze=False)
            for r, (attack, (imgs, titles)) in enumerate(rows.items()):
                for c, (img, t) in enumerate(zip(imgs, titles)):
                    axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
                    axes[r, c].set_title(t, fontsize=9)
                    axes[r, c].axis("off")
                axes[r, 0].text(-6, 14, attack.upper(), rotation=90, va="center", ha="center", fontsize=11)
            clean_pred = int(np.argmax(data["probs"]["eps_0.0"]))
            note = "" if clean_pred == digit else f" | already misclassified as {clean_pred} at ε=0"
            data["clean_correct"] = clean_pred == digit
            fig.suptitle(f"{label} (ep{args.sample_epochs}, trial 1) | test image #{idx} | true label {digit}{note}")
            fig.tight_layout(h_pad=2.0)
            fig.savefig(os.path.join(folder, f"sample_{k}.png"), dpi=100)
            plt.close(fig)
            with open(os.path.join(folder, f"sample_{k}_data.json"), "w") as f:
                json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODEL_LABELS)
    p.add_argument("--attacks", nargs="+", default=["fgsm", "pgd"], choices=["fgsm", "pgd"])
    p.add_argument("--fgsm-epochs", nargs="+", type=int, default=EPOCHS)
    p.add_argument("--pgd-epochs", nargs="+", type=int, default=[10],
                   help="PGD is ~40x the cost of FGSM, so by default it runs on the 10-epoch models only")
    p.add_argument("--pgd-steps", type=int, default=40)
    p.add_argument("--pgd-step-frac", type=float, default=0.1, help="PGD step size alpha = eps * this")
    p.add_argument("--pgd-n", type=int, default=10000, help="number of test images (from the start) for PGD")
    p.add_argument("--out", default=ATTACK_CSV)
    p.add_argument("--resume", action="store_true", help="skip rows already present in --out instead of overwriting it")
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--samples-only", action="store_true", help="only regenerate visuals/samples")
    p.add_argument("--sample-epochs", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(0)
    device = get_device()
    x, y = test_tensors(device)

    if args.samples_only:
        for label in args.models:
            save_samples(load_model(label, args.sample_epochs, 1, device), label, x, y, args)
        return

    done = load_done(args.out) if args.resume else set()
    if not args.resume or not os.path.exists(args.out):
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

    def write(row):
        with open(args.out, "a", newline="") as f:
            csv.writer(f).writerow(row)

    for label in args.models:
        for ep in sorted(set(args.fgsm_epochs) | set(args.pgd_epochs)):
            for trial in TRIALS:
                model = load_model(label, ep, trial, device)
                _, _, clean_preds = evaluate(model, x, y)
                save_cm("clean", label, ep, trial, confusion_matrix(y, clean_preds))

                if "fgsm" in args.attacks and ep in args.fgsm_epochs:
                    sign = None
                    for eps in EPSILONS:
                        if (label, ep, trial, "fgsm", eps) in done:
                            continue
                        if sign is None:
                            sign = torch.cat([input_gradient(model, x[i:i + 1000], y[i:i + 1000]).sign()
                                              for i in range(0, len(x), 1000)])
                        acc, loss, preds = evaluate(model, fgsm(model, x, y, eps, grad_sign=sign), y)
                        save_cm(f"fgsm_{eps:g}", label, ep, trial, confusion_matrix(y, preds))
                        write([label, f"{ep:02d}", trial, "fgsm", eps, len(x), round(acc, 2), round(loss, 4)])
                        print(f"{label} ep{ep:02d} t{trial} fgsm eps={eps:g}: acc={acc:.2f} loss={loss:.3f}", flush=True)

                if "pgd" in args.attacks and ep in args.pgd_epochs:
                    xs, ys = x[:args.pgd_n], y[:args.pgd_n]
                    for eps in EPSILONS:
                        if (label, ep, trial, "pgd", eps) in done:
                            continue
                        x_adv = pgd(model, xs, ys, eps, args.pgd_steps, args.pgd_step_frac,
                                    seed=zlib.crc32(f"{label}_{ep}_{trial}_{eps:g}".encode()))
                        acc, loss, preds = evaluate(model, x_adv, ys)
                        save_cm(f"pgd_{eps:g}", label, ep, trial, confusion_matrix(ys, preds))
                        write([label, f"{ep:02d}", trial, "pgd", eps, len(xs), round(acc, 2), round(loss, 4)])
                        print(f"{label} ep{ep:02d} t{trial} pgd eps={eps:g}: acc={acc:.2f} loss={loss:.3f}", flush=True)

        if not args.no_samples:
            save_samples(load_model(label, args.sample_epochs, 1, device), label, x, y, args)


if __name__ == "__main__":
    main()
