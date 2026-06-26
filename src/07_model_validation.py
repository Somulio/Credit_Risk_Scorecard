"""
Step 8: Model Validation
AUC, Gini, KS, decile/rank-ordering, PSI (train vs test, train vs OOT).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def auc_gini_ks(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = np.max(np.abs(tpr - fpr)) * 100
    return auc, gini, ks


def decile_rank_order(df: pd.DataFrame, score_col="Score", target_col=None):
    """10 score bands with bad-rate; good models show bad rate falling as score rises."""
    target_col = target_col or cfg.TARGET_COL
    d = df.copy()
    d["Decile"] = pd.qcut(d[score_col], 10, labels=False, duplicates="drop")
    rank = d.groupby("Decile").agg(
        accounts=(target_col, "size"),
        bads=(target_col, "sum"),
        avg_score=(score_col, "mean"),
    ).reset_index()
    rank["bad_rate_pct"] = (rank["bads"] / rank["accounts"] * 100).round(2)
    rank = rank.sort_values("avg_score").reset_index(drop=True)
    rank["cum_bads_pct"] = (rank["bads"].cumsum() / rank["bads"].sum() * 100).round(1)
    return rank


def psi(expected: np.ndarray, actual: np.ndarray, bins=10):
    """Population Stability Index using train-derived score buckets."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    a_pct = np.histogram(actual, breakpoints)[0] / len(actual)
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)
    return float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))


def concordance_discordance(y_true, y_score, sample_size=2000):
    """Pairwise concordant/discordant % between bad and good predicted scores."""
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    goods = y_score[y_true == 0]
    bads = y_score[y_true == 1]
    n = min(sample_size, len(goods) * len(bads))
    gi = rng.integers(0, len(goods), n)
    bi = rng.integers(0, len(bads), n)
    g, b = goods[gi], bads[bi]
    concordant = (b > g).mean() * 100
    discordant = (b < g).mean() * 100
    tied = 100 - concordant - discordant
    return concordant, discordant, tied


def run():
    results = {}
    scored = {}
    for name in ["train", "test", "oot"]:
        df = pd.read_csv(os.path.join(cfg.PROCESSED_DATA_DIR, f"{name}_scored.csv"))
        scored[name] = df
        auc, gini, ks = auc_gini_ks(df[cfg.TARGET_COL], df["PD"])
        results[name] = {"AUC": auc, "Gini": gini, "KS": ks, "n": len(df), "bad_rate": df[cfg.TARGET_COL].mean()}

    perf = pd.DataFrame(results).T
    perf.to_csv(os.path.join(cfg.REPORTS_DIR, "09_model_performance_summary.csv"))
    print("=== Model Performance (AUC / Gini / KS) ===")
    print(perf.round(4).to_string())

    for name in ["train", "test", "oot"]:
        verdict_gini = "Good" if results[name]["Gini"] >= cfg.GINI_GOOD_THRESHOLD else "Below industry benchmark"
        verdict_ks = "Excellent" if results[name]["KS"] >= cfg.KS_EXCELLENT_THRESHOLD else (
            "Good" if results[name]["KS"] >= cfg.KS_GOOD_THRESHOLD else "Below benchmark")
        print(f"{name:>5}: Gini={results[name]['Gini']*100:.1f}% [{verdict_gini}] | KS={results[name]['KS']:.1f} [{verdict_ks}]")

    # Rank ordering / decile analysis on TEST
    decile = decile_rank_order(scored["test"])
    decile.to_csv(os.path.join(cfg.REPORTS_DIR, "10_decile_rank_order_test.csv"), index=False)
    print("\n=== Decile Rank Order (TEST) ===")
    print(decile.to_string(index=False))
    monotonic = (decile["bad_rate_pct"].diff().dropna() <= 0.5).mean() >= 0.7
    print(f"Rank ordering monotonic (bad rate falls as score rises): {'PASS' if monotonic else 'REVIEW'}")

    # PSI: train (expected/development) vs test, train vs OOT
    psi_test = psi(scored["train"]["Score"].values, scored["test"]["Score"].values)
    psi_oot = psi(scored["train"]["Score"].values, scored["oot"]["Score"].values)
    print(f"\nPSI Train vs Test: {psi_test:.4f}  | PSI Train vs OOT: {psi_oot:.4f}")
    for label, val in [("Train vs Test", psi_test), ("Train vs OOT", psi_oot)]:
        verdict = "Stable" if val < cfg.PSI_STABLE_THRESHOLD else ("Monitor" if val < cfg.PSI_MONITOR_THRESHOLD else "Rebuild Model")
        print(f"  {label}: {verdict}")

    # Concordance on test
    c, d_, t = concordance_discordance(scored["test"][cfg.TARGET_COL].values, scored["test"]["PD"].values)
    print(f"\nConcordant: {c:.1f}% | Discordant: {d_:.1f}% | Tied: {t:.1f}%")

    summary = {
        "performance": perf,
        "decile": decile,
        "psi_test": psi_test,
        "psi_oot": psi_oot,
        "concordance": (c, d_, t),
    }
    return summary


if __name__ == "__main__":
    run()
