import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluation, scorecard


def test_gini_and_ks_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    metrics = evaluation.evaluate(y, score)
    assert metrics["AUC_ROC"] == 1.0
    assert metrics["Gini"] == 1.0
    assert metrics["KS"] == 1.0


def test_psi_identical_distributions_is_zero():
    s = pd.Series(np.random.RandomState(0).normal(size=500))
    assert evaluation.psi(s, s) < 1e-6


def test_score_from_proba_monotonic_with_risk():
    low_risk = scorecard.score_from_proba(np.array([0.01]))
    high_risk = scorecard.score_from_proba(np.array([0.5]))
    assert low_risk[0] > high_risk[0]
