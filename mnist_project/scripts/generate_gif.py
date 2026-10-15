"""Animate the row-normalised confusion matrix (mean of 3 trials) as epsilon increases.

Usage: python mnist_project/scripts/generate_gif.py --model simple_mlp --attack fgsm --epochs 10
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

from common import CM_DIR, DISPLAY_NAMES, EPSILONS, MODEL_LABELS, TRIALS, VISUALS_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="simple_mlp", choices=MODEL_LABELS)
    p.add_argument("--attack", default="fgsm", choices=["fgsm", "pgd"])
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(eps):
        ax.clear()
        folder = "clean" if eps == 0.0 else f"{args.attack}_{eps:g}"
        cms = [np.load(path) for t in TRIALS
               if os.path.exists(path := os.path.join(CM_DIR, folder, f"{args.model}_ep{args.epochs:02d}_t{t}_cm.npy"))]
        if not cms:
            ax.text(0.5, 0.5, f"Missing data: {folder}", ha="center")
            return
        avg = np.mean(cms, axis=0)
        sns.heatmap(avg / avg.sum(axis=1, keepdims=True), cmap="Blues", cbar=False, square=True,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f"{DISPLAY_NAMES[args.model]} | {args.attack.upper()} ε = {eps:g}", fontsize=18, pad=20)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    ani = FuncAnimation(fig, update, frames=[0.0] + EPSILONS, repeat=True, interval=500)
    out_dir = os.path.join(VISUALS_DIR, "gifs")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if args.attack == "fgsm" else f"_{args.attack}"
    out = os.path.join(out_dir, f"{args.model}{suffix}.gif")
    ani.save(out, writer="pillow", fps=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
