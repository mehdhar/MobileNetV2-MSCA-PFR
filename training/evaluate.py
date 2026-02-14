import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from torch.utils.data import DataLoader
from torchvision import transforms

from models.msca_pfr_model import MobileNetV2_MSCA_PFR
from utils.dataset_loader import ImageFolderDataset


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_percent = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = cm[i, j]
            p = cm_percent[i, j]
            annot[i, j] = f"{p:.1f}%\n{c}"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate():

    # -----------------------------
    # Load configuration
    # -----------------------------
    config = load_config("config.yaml")

    val_dir = config["paths"]["val_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    num_classes = config["model"]["num_classes"]

    # -----------------------------
    # Device setup
    # -----------------------------
    use_gpu = config["device"]["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    print(f"\nEvaluating on device: {device}\n")

    # -----------------------------
    # Load dataset
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = ImageFolderDataset(val_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    class_names = val_dataset.classes

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = MobileNetV2_MSCA_PFR(num_classes=num_classes).to(device)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError("best_model.pth not found!")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # -----------------------------
    # Evaluation loop
    # -----------------------------
    all_labels = []
    all_preds = []
    all_probs = []

    import time
    start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = (time.time() - start_time) / len(val_dataset)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # -----------------------------
    # Compute Metrics
    # -----------------------------
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    mcc = matthews_corrcoef(all_labels, all_preds)

    # -----------------------------
    # Save confusion matrix
    # -----------------------------
    plot_confusion_matrix(
        cm, 
        class_names,
        os.path.join(checkpoint_dir, "confusion_matrix.png")
    )

    # -----------------------------
    # ROC Curve
    # -----------------------------
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "roc_curve.png"))
    plt.close()

    # -----------------------------
    # Precision-Recall Curve
    # -----------------------------
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_true_bin[:, i], all_probs[:, i]
        )
        plt.plot(recall_vals, precision_vals, label=f"{class_names[i]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "precision_recall_curve.png"))
    plt.close()

    # -----------------------------
    # Save metric summary
    # -----------------------------
    metrics_path = os.path.join(checkpoint_dir, "evaluation_report.txt")
    with open(metrics_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"MCC:       {mcc:.4f}\n")

    # Inference time
    inference_path = os.path.join(checkpoint_dir, "inference_time.txt")
    with open(inference_path, "w") as f:
        f.write(f"Inference time per image: {inference_time:.6f} seconds\n")

    print("\nEvaluation complete!")
    print(f"Results saved to: {checkpoint_dir}\n")


if __name__ == "__main__":
    evaluate()
