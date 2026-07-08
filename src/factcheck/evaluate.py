"""Evaluation: macro-average F1 (high class imbalance in dataset), per the CLIMATE-FEVER protocol."""

from sklearn.metrics import classification_report, f1_score

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro") 

def full_report(y_true, y_pred):
    """Per-class precision/recall/F1 macro average, as a string."""
    return classification_report(y_true, y_pred, zero_division=0)