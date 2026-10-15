"""Show one confusion matrix, or the mean over the 3 trials with --avg.

Usage: python mnist_project/scripts/view_confusion_matrix.py --attack fgsm --eps 0.05 --model simple_mlp --epochs 10 --trial 2
       python mnist_project/scripts/view_confusion_matrix.py --attack pgd --eps 0.15 --model robust_cnn --avg
Use --eps 0 for the clean test set.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import CM_DIR, MODEL_LABELS, TRIALS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attack", default="fgsm", choices=["fgsm", "pgd"])
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--model", default="simple_mlp", choices=MODEL_LABELS)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--avg", action="store_true", help="average the 3 trials")
    p.add_argument("--save", help="save to this path instead of showing a window")
    args = p.parse_args()

    folder = "clean" if args.eps == 0 else f"{args.attack}_{args.eps:g}"
    trials = TRIALS if args.avg else [args.trial]
    paths = [os.path.join(CM_DIR, folder, f"{args.model}_ep{args.epochs:02d}_t{t}_cm.npy") for t in trials]
    missing = [pth for pth in paths if not os.path.exists(pth)]
    if missing:
        raise SystemExit("Not found:\n  " + "\n  ".join(missing) + "\nRun scripts/run_attacks.py first.")

    cm = np.mean([np.load(pth) for pth in paths], axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".1f" if args.avg else ".0f", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    which = f"mean of {len(paths)} trials" if args.avg else f"trial {args.trial}"
    plt.title(f"Confusion matrix: {args.model} | {folder} | {args.epochs} epochs | {which}")
    if args.save:
        plt.savefig(args.save, dpi=100)
    else:
        plt.show()


if __name__ == "__main__":
    main()
