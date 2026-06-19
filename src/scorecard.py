"""Convert logistic-regression log-odds into a points-based scorecard."""
import numpy as np
from . import config


def get_factor_offset():
    factor = config.SCORECARD_PDO / np.log(2)
    offset = config.SCORECARD_BASE_SCORE - factor * np.log(config.SCORECARD_BASE_ODDS)
    return factor, offset


def score_from_proba(p_bad: np.ndarray) -> np.ndarray:
    """Map predicted PD to a credit score (higher score = lower risk)."""
    p_bad = np.clip(p_bad, 1e-6, 1 - 1e-6)
    log_odds_good = np.log((1 - p_bad) / p_bad)
    factor, offset = get_factor_offset()
    return offset + factor * log_odds_good


def build_points_table(model, feature_names: list) -> dict:
    """Allocate scorecard points per WOE-coefficient for documentation."""
    factor, _ = get_factor_offset()
    coefs = dict(zip(feature_names, model.coef_[0]))
    return {f: round(-coefs[f] * factor, 2) for f in coefs}
