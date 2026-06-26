# Methodology Notes — Variable Selection & Overfitting Investigation

This file documents a real issue hit during development, kept here deliberately
because walking through it is more useful in an interview than pretending the
first model run was the final one.

## Iteration 1 — naive IV filter (0.02 < IV < 0.5)

All 24 features clearing a standard IV band were fed straight into logistic
regression.

| Sample | Gini | KS |
|---|---|---|
| Train | 83.2% | 69.0 |
| Test | **10.1%** | **12.9** |
| OOT | 37.9% | 44.1 |

A train-test gap this large (83% → 10%) is a classic overfitting signature:
24 variables, many with thin bins, fit against only ~62 bad accounts in the
training fold. The model was memorizing train-specific noise in low-population
bins rather than learning a generalizable relationship.

**Fixes applied:**
1. Capped `OptimalBinning` to `max_n_bins=4`, `min_bin_size=0.10` — fewer,
   larger bins are inherently less prone to overfitting on a small bad count.
2. Replaced "keep everything that survives IV/corr/VIF" with **greedy forward
   stepwise selection** using 5-fold CV AUC, capped at 12 variables — adds one
   variable at a time only if it improves cross-validated performance.
3. Added L2 regularization (`C=0.5`) to the logistic regression itself.

## Iteration 2 — six implausibly strong IV variables included

Relaxing the IV ceiling to let in `Negative_Balance_Days`, `Avg_Balance_3M/6M`,
bureau inquiry counts, `DPD_30`, `Credit_Card_Utilization` pushed performance to:

| Sample | Gini | KS |
|---|---|---|
| Train | 96.7% | 90.4 |
| Test | 92.3% | 90.0 |
| OOT | 96.7% | 97.1 |

Consistent across samples (good), but **implausibly high for a real retail
credit model** — Gini above ~85% on a held-out OOT sample almost always means
a feature is functioning as a near-proxy for the target rather than a genuine
predictor. Checking the raw distributions confirmed it:

```
Negative_Balance_Days:  bad mean 17.4 (std 8.1)  vs  good mean 2.7 (std 3.9)
                        -> almost no overlap between the two populations
```

This is a hallmark of synthetic-data generation (the field was likely used,
directly or indirectly, to construct the `Default` label), not a real-world
behavioral relationship — in real bureau data even the strongest single
predictor rarely produces this little overlap.

**Fix applied:** these six fields were treated as "pending SME review" and
excluded from this development cycle (`config.IV_SUSPICIOUS_THRESHOLD = 1.5`),
keeping only the moderately strong, plausibly-overlapping fields
(`Income_INR`, `Savings_Account_Balance`, `Outstanding_Loans`) on an explicit
business-approved list (`config.BUSINESS_APPROVED_HIGH_IV_VARS`).

## Final result

| Sample | Gini | KS |
|---|---|---|
| Train | 92.4% | 84.3 |
| Test | 77.6% | 65.6 |
| OOT | 84.3% | 75.5 |

Still strong — stronger than a typical real bank scorecard (Gini 76% / KS 69%
is roughly what production scorecards in this space tend to land on)  — but the
train→test→OOT consistency is the important signal: no >15pt Gini drop between
samples, monotonic rank ordering, stable PSI. The honest caveat (documented in
the validation report) is that this dataset is synthetic and likely still has
some baked-in separability beyond what 1,000 real accounts would show; a
production sign-off would require re-running this same pipeline against a
larger, real, less-separable portfolio.

**Takeaway for interviews:** the point of this exercise isn't "I got Gini
0.92" — it's "Gini 0.92 made me suspicious, here's how I checked it, and here's
why I didn't ship it." That diagnostic instinct is what model validation teams
are actually screening for.
