import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


# -----------------------------
# Confusion Matrix Plot
# -----------------------------
def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plots and saves a clean confusion matrix with percentages.
    """

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]

            if i == j:
                annot[i, j] = f"{p:.1f}%\n{c}/{cm_sum[i][0]}"
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{p:.1f}%\n{c}"

    cm_df = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Number of Predictions"},
        linewidths=0.5,
        linecolor="gray"
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# ROC Curve Plot
# -----------------------------
def plot_roc_curves(fpr_dict, tpr_dict, auc_dict, save_path):
    """
    Plots ROC curves for all classes.
    """

    plt.figure(figsize=(10, 7))

    for cls, fpr in fpr_dict.items():
        tpr = tpr_dict[cls]
        roc_auc = auc_dict[cls]
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# Precision-Recall Curve Plot
# -----------------------------
def plot_pr_curves(precision_dict, recall_dict, class_names, save_path):
    """
    Precision-Recall curves + iso-F1 lines.
    """

    plt.figure(figsize=(10, 7))

    colors = cycle(["navy", "darkorange", "green", "purple", "red", "blue"])

    # Iso-F1 lines
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f in f_scores:
        x = np.linspace(0.01, 1)
        y = f * x / (2 * x - f)
        y = y[y >= 0]
        x = x[: len(y)]
        plt.plot(x, y, color="gray", alpha=0.2)
        plt.annotate(f"f1={f:.1f}", xy=(0.9, y[-1] + 0.02))

    # Plot curves
    for cls, color in zip(class_names, colors):
        precision = precision_dict[cls]
        recall = recall_dict[cls]
        plt.plot(recall, precision, lw=2, color=color, label=cls)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
