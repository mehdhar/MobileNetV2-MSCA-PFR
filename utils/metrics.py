import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize


def compute_basic_metrics(labels, predictions):
    """
    Returns accuracy, precision, recall, f1-score (weighted).
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")

    return accuracy, precision, recall, f1


def compute_classification_report(labels, predictions, class_names):
    """
    Returns a formatted classification report string.
    """
    return classification_report(labels, predictions, target_names=class_names, digits=4)


def compute_confusion_matrix(labels, predictions):
    """
    Returns the confusion matrix.
    """
    return confusion_matrix(labels, predictions)


def compute_mcc(labels, predictions):
    """
    Returns Matthews Correlation Coefficient.
    """
    return matthews_corrcoef(labels, predictions)


def compute_roc_curves(labels, probs, class_names):
    """
    Computes ROC curves for all classes.
    
    Returns:
    - fpr_dict[class_name]
    - tpr_dict[class_name]
    - auc_dict[class_name]
    """

    num_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=np.arange(num_classes))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], np.array(probs)[:, i])
        roc_auc = auc(fpr, tpr)

        fpr_dict[cls] = fpr
        tpr_dict[cls] = tpr
        auc_dict[cls] = roc_auc

    return fpr_dict, tpr_dict, auc_dict


def compute_pr_curves(labels, probs, class_names):
    """
    Computes Precision-Recall curves for all classes.
    
    Returns:
    - precision_dict
    - recall_dict
    """

    num_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=np.arange(num_classes))

    precision_dict, recall_dict = {}, {}

    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(labels_bin[:, i], np.array(probs)[:, i])
        precision_dict[cls] = precision
        recall_dict[cls] = recall

    return precision_dict, recall_dict
