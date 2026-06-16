import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve


def link_loss(scores, labels):
    labels = labels.float()
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    losses = []
    if pos_scores.numel() > 0:
        losses.append(F.logsigmoid(pos_scores))
    if neg_scores.numel() > 0:
        losses.append(F.logsigmoid(-neg_scores))
    return -torch.mean(torch.cat(losses)) if losses else scores.new_tensor(0.0)


def binary_auc(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    neg_sorted = np.sort(neg)
    higher = np.searchsorted(neg_sorted, pos, side="left").sum()
    higher_or_equal = np.searchsorted(neg_sorted, pos, side="right").sum()
    return float((higher + 0.5 * (higher_or_equal - higher)) / (pos.size * neg.size))


def compute_metrics(scores, labels):
    y_scores = scores.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(np.int64)
    y_pred = np.zeros_like(y_true, dtype=np.int64)
    true_num = int(y_true.sum())
    if true_num > 0:
        topk = min(true_num, y_scores.shape[0])
        y_pred[np.argpartition(-y_scores, topk - 1)[:topk]] = 1
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return {
        "loss": None,
        "auc": binary_auc(y_true, y_scores),
        "pr_auc": float(auc(rs, ps)),
        "f1": float(f1_score(y_true, y_pred)),
    }
