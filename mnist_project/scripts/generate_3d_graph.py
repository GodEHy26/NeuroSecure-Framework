"""Plot accuracy (or loss) under FGSM as a surface over (epsilon, training epochs).

The measured grid is 10 epoch budgets x 10 epsilons (each point the mean of 3 trials).
The surface between grid points is filled in by cubic interpolation (scipy griddata), so
it is a smoothed visual aid, not measured data; measured points are drawn as black dots.

Usage: python mnist_project/scripts/generate_3d_graph.py [--models ...] [--metric Adv_Accuracy|Adv_Loss]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from common import ATTACK_CSV, DISPLAY_NAMES, MODEL_LABELS, VISUALS_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODEL_LABELS)
    p.add_argument("--attack", default="fgsm", help="only FGSM is run at every epoch budget by default")
    p.add_argument("--metric", default="Adv_Accuracy", choices=["Adv_Accuracy", "Adv_Loss"])
    args = p.parse_args()

    out_dir = os.path.join(VISUALS_DIR, "3d_graphs")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(ATTACK_CSV)
    df = df[df["Attack"] == args.attack]

    for label in args.models:
        pivot = df[df["Model_Type"] == label].pivot_table(index="Epochs", columns="Epsilon",
                                                          values=args.metric, aggfunc="mean")
        if pivot.index.nunique() < 2:
            print(f"Skipping {label}: {args.attack} was not run at more than one epoch budget")
            continue
        eps_dense = np.linspace(pivot.columns.min(), pivot.columns.max(), 50)
        ep_dense = np.linspace(pivot.index.min(), pivot.index.max(), 50)
        X, Y = np.meshgrid(eps_dense, ep_dense)
        pts = [(eps, ep) for ep in pivot.index for eps in pivot.columns]
        vals = pivot.values.flatten()
        Z = griddata(pts, vals, (X, Y), method="cubic")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.85, antialiased=True)
        ax.scatter([e for e, _ in pts], [ep for _, ep in pts], vals, color="black", s=6, label="measured (mean of 3 trials)")
        ax.view_init(elev=30, azim=-60)
        metric_name = "Accuracy (%)" if args.metric == "Adv_Accuracy" else "Cross-entropy loss"
        ax.set_title(f"{DISPLAY_NAMES[label]}: {metric_name} under {args.attack.upper()}\n"
                     "surface = cubic interpolation between measured points")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Training epochs")
        ax.set_zlabel(metric_name)
        ax.legend(loc="upper right")
        suffix = "" if args.metric == "Adv_Accuracy" else "_loss"
        path = os.path.join(out_dir, f"3d_smooth_{label}{suffix}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
