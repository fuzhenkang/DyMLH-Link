import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def binary_metrics(logits, labels):
    labels_np = labels.detach().cpu().numpy()
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
    }
    if len(np.unique(labels_np)) > 1:
        metrics["auc"] = float(roc_auc_score(labels_np, probs))
        metrics["pr_auc"] = float(average_precision_score(labels_np, probs))
    return metrics
