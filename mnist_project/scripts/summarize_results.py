"""Aggregate the raw results into the tables and figures used in the paper.

Every number is reported as mean ± standard deviation over the 3 independent trials
(sample standard deviation, ddof=1). The headline numbers use the 10-epoch models;
the all-epoch FGSM averages are also written so older drafts can be cross-checked.

Writes:
  results/summary_clean.csv          clean accuracy / loss per model and epoch budget
  results/summary_attacks.csv        adversarial accuracy / loss per model, attack, epoch budget, epsilon
  results/summary_predictions.csv    share of test images predicted as each class (10-epoch models)
  results/summary.md                 the paper tables in Markdown
  visuals/figures/fig*.png           paper figures
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ATTACK_CSV, CM_DIR, DISPLAY_NAMES, EPSILONS, FIGURES_DIR, MODEL_LABELS, RESULTS_DIR,
                    TRAINING_CSV, TRIALS)

COLORS = {"simple_mlp": "#2a78d6", "simple_cnn": "#eb6834", "robust_cnn": "#1baf7a"}
MARKERS = {"simple_mlp": "o", "simple_cnn": "s", "robust_cnn": "^"}
FINAL_EPOCH = 10


def pm(mean, std, digits=1):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def prediction_shares(attack, eps, label):
    folder = "clean" if eps == 0 else f"{attack}_{eps:g}"
    cms = [np.load(os.path.join(CM_DIR, folder, f"{label}_ep{FINAL_EPOCH:02d}_t{t}_cm.npy")) for t in TRIALS]
    shares = [cm.sum(axis=0) / cm.sum() * 100 for cm in cms]       # column sums = predicted-class counts
    return np.mean(shares, axis=0), np.std(shares, axis=0, ddof=1)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    train = pd.read_csv(TRAINING_CSV)
    att = pd.read_csv(ATTACK_CSV)

    clean = (train.groupby(["Model_Type", "Epochs"])
             .agg(Acc_Mean=("Accuracy", "mean"), Acc_Std=("Accuracy", "std"),
                  Loss_Mean=("Test_Loss", "mean"), Loss_Std=("Test_Loss", "std"),
                  Loss_Min=("Test_Loss", "min"), Loss_Max=("Test_Loss", "max"))
             .reset_index())
    clean.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_clean.csv"), index=False)

    summ = (att.groupby(["Model_Type", "Attack", "Epochs", "Epsilon"])
            .agg(N_Images=("N_Images", "first"),
                 Acc_Mean=("Adv_Accuracy", "mean"), Acc_Std=("Adv_Accuracy", "std"),
                 Acc_Min=("Adv_Accuracy", "min"), Acc_Max=("Adv_Accuracy", "max"),
                 Loss_Mean=("Adv_Loss", "mean"), Loss_Std=("Adv_Loss", "std"),
                 Loss_Min=("Adv_Loss", "min"), Loss_Max=("Adv_Loss", "max"))
            .reset_index())
    summ.round(4).to_csv(os.path.join(RESULTS_DIR, "summary_attacks.csv"), index=False)

    def get(label, attack, ep=FINAL_EPOCH):
        return summ[(summ.Model_Type == label) & (summ.Attack == attack) & (summ.Epochs == ep)].set_index("Epsilon")

    attacks = [a for a in ["fgsm", "pgd"] if a in set(att.Attack)]
    md = ["# Results summary", "",
          f"All values are mean ± standard deviation over {len(TRIALS)} independently trained models "
          f"(trials). Unless stated otherwise the models were trained for {FINAL_EPOCH} epochs.", ""]

    # Table 1: clean baseline
    md += ["## Table 1. Clean test-set performance (10-epoch models)", "",
           "| Model | Clean accuracy (%) | Test loss |", "|---|---|---|"]
    for label in MODEL_LABELS:
        r = clean[(clean.Model_Type == label) & (clean.Epochs == FINAL_EPOCH)].iloc[0]
        md.append(f"| {DISPLAY_NAMES[label]} | {pm(r.Acc_Mean, r.Acc_Std, 2)} | {pm(r.Loss_Mean, r.Loss_Std, 3)} |")
    md.append("")

    # Tables 2-5: accuracy and loss per attack
    for attack in attacks:
        n = int(get(MODEL_LABELS[0], attack).N_Images.iloc[0])
        for metric, name, digits in [("Acc", "accuracy (%)", 1), ("Loss", "cross-entropy loss", 2)]:
            md += [f"## {attack.upper()} {name}, 10-epoch models ({n:,} test images)", "",
                   "| ε | " + " | ".join(DISPLAY_NAMES[m] for m in MODEL_LABELS) + " |",
                   "|---|" + "---|" * len(MODEL_LABELS)]
            for eps in EPSILONS:
                cells = [pm(get(m, attack).loc[eps, f"{metric}_Mean"], get(m, attack).loc[eps, f"{metric}_Std"], digits)
                         for m in MODEL_LABELS]
                md.append(f"| {eps:g} | " + " | ".join(cells) + " |")
            md.append("")

    if "pgd" in attacks:
        md += ["## PGD vs FGSM at the same ε (10-epoch models, accuracy %)", "",
               "| ε | " + " | ".join(f"{DISPLAY_NAMES[m]} FGSM / PGD" for m in MODEL_LABELS) + " |",
               "|---|" + "---|" * len(MODEL_LABELS)]
        for eps in EPSILONS:
            md.append(f"| {eps:g} | " + " | ".join(
                f"{get(m, 'fgsm').loc[eps, 'Acc_Mean']:.1f} / {get(m, 'pgd').loc[eps, 'Acc_Mean']:.1f}"
                for m in MODEL_LABELS) + " |")
        md.append("")

    # Trial spread
    md += ["## Spread between trials (FGSM accuracy %, 10-epoch models: min – max of 3 trials)", "",
           "| ε | " + " | ".join(DISPLAY_NAMES[m] for m in MODEL_LABELS) + " |", "|---|" + "---|" * len(MODEL_LABELS)]
    for eps in EPSILONS:
        md.append(f"| {eps:g} | " + " | ".join(
            f"{get(m, 'fgsm').loc[eps, 'Acc_Min']:.1f} – {get(m, 'fgsm').loc[eps, 'Acc_Max']:.1f}" for m in MODEL_LABELS) + " |")
    md.append("")

    # Epoch trend
    md += ["## FGSM accuracy (%) by training epochs (mean of 3 trials)", "",
           "| Model | ε | " + " | ".join(f"ep{e}" for e in range(1, 11)) + " |", "|---|---|" + "---|" * 10]
    for label, eps in [("simple_mlp", 0.05), ("simple_mlp", 0.1), ("simple_cnn", 0.1), ("simple_cnn", 0.5),
                       ("robust_cnn", 0.05), ("robust_cnn", 0.5)]:
        vals = [summ[(summ.Model_Type == label) & (summ.Attack == "fgsm") & (summ.Epochs == e) & (summ.Epsilon == eps)]
                .Acc_Mean.iloc[0] for e in range(1, 11)]
        md.append(f"| {DISPLAY_NAMES[label]} | {eps:g} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
    md.append("")
    md += ["## FGSM loss by training epochs (mean of 3 trials)", "",
           "| Model | ε | " + " | ".join(f"ep{e}" for e in range(1, 11)) + " |", "|---|---|" + "---|" * 10]
    for label, eps in [("simple_mlp", 0.1), ("simple_cnn", 0.1), ("simple_cnn", 0.5), ("robust_cnn", 0.1),
                       ("robust_cnn", 0.5)]:
        vals = [summ[(summ.Model_Type == label) & (summ.Attack == "fgsm") & (summ.Epochs == e) & (summ.Epsilon == eps)]
                .Loss_Mean.iloc[0] for e in range(1, 11)]
        md.append(f"| {DISPLAY_NAMES[label]} | {eps:g} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    md.append("")

    # All-epoch averages (what the first draft reported)
    md += ["## FGSM accuracy (%) averaged over all 90 models (epochs 1–10, 3 trials each)", "",
           "Provided only for cross-checking earlier drafts; the paper should use the 10-epoch tables.", "",
           "| ε | " + " | ".join(DISPLAY_NAMES[m] for m in MODEL_LABELS) + " |", "|---|" + "---|" * len(MODEL_LABELS)]
    fg = att[att.Attack == "fgsm"].groupby(["Model_Type", "Epsilon"]).Adv_Accuracy.mean()
    for eps in EPSILONS:
        md.append(f"| {eps:g} | " + " | ".join(f"{fg.loc[(m, eps)]:.2f}" for m in MODEL_LABELS) + " |")
    md.append("")

    # Prediction distribution
    rows = []
    for attack in attacks:
        for label in MODEL_LABELS:
            for eps in [0.0] + EPSILONS:
                mean, std = prediction_shares(attack, eps, label)
                rows.append({"Attack": attack, "Model_Type": label, "Epsilon": eps,
                             **{f"Pred_{c}_Mean": round(mean[c], 2) for c in range(10)},
                             **{f"Pred_{c}_Std": round(std[c], 2) for c in range(10)}})
    preds = pd.DataFrame(rows)
    preds.to_csv(os.path.join(RESULTS_DIR, "summary_predictions.csv"), index=False)
    md += ["## Share of all test images predicted as each digit (%), 10-epoch models", "",
           "Under attack every image is pushed away from its true label; the column shows where the predictions go.", ""]
    for attack in attacks:
        md += [f"### {attack.upper()}", "", "| Model | ε | most-predicted digit | share (%) | share predicted as 8 (%) |",
               "|---|---|---|---|---|"]
        for label in MODEL_LABELS:
            for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                r = preds[(preds.Attack == attack) & (preds.Model_Type == label) & (preds.Epsilon == eps)].iloc[0]
                means = [r[f"Pred_{c}_Mean"] for c in range(10)]
                top = int(np.argmax(means))
                md.append(f"| {DISPLAY_NAMES[label]} | {eps:g} | {top} | {pm(means[top], r[f'Pred_{top}_Std'])} | "
                          f"{pm(r['Pred_8_Mean'], r['Pred_8_Std'])} |")
        md.append("")

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(md))

    # ---------- Figures ----------
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

    # Figure 1: clean accuracy and loss vs epoch budget
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label in MODEL_LABELS:
        c = clean[clean.Model_Type == label]
        for ax, col in zip(axes, ["Acc", "Loss"]):
            ax.errorbar(c.Epochs, c[f"{col}_Mean"], yerr=c[f"{col}_Std"], color=COLORS[label], marker=MARKERS[label],
                        markersize=6, linewidth=2, capsize=3, label=DISPLAY_NAMES[label])
    axes[0].set(xlabel="Training epochs", ylabel="Clean test accuracy (%)", title="Clean accuracy")
    axes[1].set(xlabel="Training epochs", ylabel="Clean test loss", title="Clean cross-entropy loss")
    for ax in axes:
        style(ax)
        ax.set_xticks(range(1, 11))
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_clean_baseline.png"), dpi=150)
    plt.close(fig)

    # Figures 2/3: accuracy and loss vs epsilon for each attack (10-epoch models)
    for metric, ylabel, fname, logy in [("Acc", "Accuracy under attack (%)", "fig2_accuracy_vs_epsilon.png", False),
                                        ("Loss", "Cross-entropy loss under attack", "fig3_loss_vs_epsilon.png", True)]:
        fig, axes = plt.subplots(1, len(attacks), figsize=(5.5 * len(attacks), 4.2), sharey=True, squeeze=False)
        for ax, attack in zip(axes[0], attacks):
            for label in MODEL_LABELS:
                g = get(label, attack)
                x = [0.0] + list(g.index)
                r = clean[(clean.Model_Type == label) & (clean.Epochs == FINAL_EPOCH)].iloc[0]
                y = [r[f"{metric}_Mean"]] + list(g[f"{metric}_Mean"])
                if logy:   # +/- std can go below zero on a log axis, so show the min-max range of the trials
                    lo = [r[f"{metric}_Mean"] - r[f"{metric}_Min"]] + list(g[f"{metric}_Mean"] - g[f"{metric}_Min"])
                    hi = [r[f"{metric}_Max"] - r[f"{metric}_Mean"]] + list(g[f"{metric}_Max"] - g[f"{metric}_Mean"])
                    e = [lo, hi]
                else:
                    e = [r[f"{metric}_Std"]] + list(g[f"{metric}_Std"])
                ax.errorbar(x, y, yerr=e, color=COLORS[label], marker=MARKERS[label], markersize=6, linewidth=2,
                            capsize=3, label=DISPLAY_NAMES[label])
            n = int(get(MODEL_LABELS[0], attack).N_Images.iloc[0])
            ax.set(xlabel="ε (l∞ perturbation budget)", title=f"{attack.upper()} ({n:,} test images)")
            if logy:
                ax.set_yscale("log")
            style(ax)
        axes[0][0].set_ylabel(ylabel)
        fig.text(0.99, 0.01, "points: mean of 3 trials; bars: " + ("min–max of trials" if logy else "± 1 std"),
                 ha="right", va="bottom", fontsize=9, color="#555555")
        axes[0][0].legend(frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        plt.close(fig)

    # Figure 4: FGSM accuracy vs training epochs at two epsilons
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, eps in zip(axes, [0.05, 0.5]):
        for label in MODEL_LABELS:
            g = summ[(summ.Model_Type == label) & (summ.Attack == "fgsm") & (summ.Epsilon == eps)]
            ax.errorbar(g.Epochs, g.Acc_Mean, yerr=g.Acc_Std, color=COLORS[label], marker=MARKERS[label],
                        markersize=6, linewidth=2, capsize=3, label=DISPLAY_NAMES[label])
        ax.set(xlabel="Training epochs", ylabel="FGSM accuracy (%)", title=f"FGSM, ε = {eps:g}")
        ax.set_xticks(range(1, 11))
        style(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_fgsm_accuracy_vs_epochs.png"), dpi=150)
    plt.close(fig)

    # Figure 5: share of predictions per digit for the Robust CNN under FGSM
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.2
    shown = [0.0, 0.1, 0.2, 0.4]
    shades = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]
    for i, eps in enumerate(shown):
        mean, std = prediction_shares("fgsm", eps, "robust_cnn")
        ax.bar(np.arange(10) + (i - 1.5) * width, mean, width * 0.9, yerr=std, color=shades[i], capsize=2,
               label=f"ε = {eps:g}", edgecolor="white", linewidth=0.5)
    ax.set(xticks=range(10), xlabel="Predicted digit", ylabel="Share of test images (%)",
           title="Robust CNN (10 epochs) under FGSM: where the predictions go")
    style(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_robust_cnn_prediction_shares.png"), dpi=150)
    plt.close(fig)

    print(f"Wrote summaries to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
