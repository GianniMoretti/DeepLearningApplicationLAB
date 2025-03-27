# metrics.py
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def accuracy(outputs, targets):
    # Calculates the maximum prediction for each sample
    _, predicted = outputs.max(1)
    return (predicted.eq(targets).sum().item() / targets.size(0))

def f1(outputs, targets):
    _, predicted = outputs.max(1)
    predicted_np = predicted.cpu().numpy()
    targets_np = targets.cpu().numpy()
    return f1_score(targets_np, predicted_np, average='macro')

def precision(outputs, targets):
    _, predicted = outputs.max(1)
    predicted_np = predicted.cpu().numpy()
    targets_np = targets.cpu().numpy()
    return precision_score(targets_np, predicted_np, average='macro')

def recall(outputs, targets):
    _, predicted = outputs.max(1)
    predicted_np = predicted.cpu().numpy()
    targets_np = targets.cpu().numpy()
    return recall_score(targets_np, predicted_np, average='macro')

def auc(outputs, targets):
    """
    Calculates the AUC (Area Under the Curve) for binary classification.
    We assume that `outputs` contains logits or probabilities for each class.
    If outputs has shape (N, 2), we use the probability for class 1.
    """
    outputs_np = outputs.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()

    # If outputs has two columns, consider the second column after softmax
    if outputs_np.ndim > 1 and outputs_np.shape[1] == 2:
        prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    else:
        # Alternatively, if outputs is one-dimensional, assume it is already the probability for the positive class
        prob = torch.sigmoid(outputs).cpu().numpy()
    try:
        auc_val = roc_auc_score(targets_np, prob)
    except Exception as e:
        print("Error calculating AUC:", e)
        auc_val = 0.0
    return auc_val

# Dictionary to select the metric based on a string
metrics = {
    "accuracy": accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "auc": auc,
}
