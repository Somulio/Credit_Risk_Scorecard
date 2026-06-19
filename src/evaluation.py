"""Model evaluation: AUC-ROC, Gini, KS statistic, PSI drift."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y_true, y_score) -> float:
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(y_true, y_score) -> dict:
    auc = roc_auc_score(y_true, y_score)
    return {
        "AUC_ROC": round(auc, 4),
        "Gini": round(2 * auc - 1, 4),
        "KS": round(ks_statistic(y_true, y_score), 4),
    }


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_pct = pd.cut(expected, edges).value_counts(normalize=True).sort_index()
    act_pct = pd.cut(actual, edges).value_counts(normalize=True).sort_index()
    exp_pct = exp_pct.replace(0, 1e-6)
    act_pct = act_pct.replace(0, 1e-6)
    return float(((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum())
