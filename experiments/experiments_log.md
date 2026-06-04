# Experiments Log: Focal Loss vs Cross-Entropy on CIFAR-10 (Small CNN)

_All runs on Modal A100-40GB, small CNN (305,706 params), 30 epochs, AdamW + cosine, batch 128._
_Append-only narrative; the structured ledger is `results.jsonl`._

## exp00 — smoke test (CE, balanced, 2k subset, 1 epoch)
Pipeline validation only. Returned valid metrics on `cuda` (acc 0.163 as expected for 1 epoch on
2k images). Confirmed: image build, CIFAR download to Volume, training, eval (acc/ECE/F1/worst-class),
Volume persistence, and JSON return all work. Proceeded to real runs.

## Balanced CIFAR-10 (exp01–exp04)
| exp | loss | γ | test acc | ECE | macro-F1 | worst-class acc | train min |
|-----|------|---|----------|-----|----------|-----------------|-----------|
| exp01 | CE | – | **0.8790** | 0.0337 | **0.8786** | 0.729 | 3.57 |
| exp02 | focal | 1 | 0.8744 | **0.0154** | 0.8741 | 0.743 | 4.17 |
| exp03 | focal | 2 | 0.8728 | 0.0660 | 0.8724 | 0.715 | 3.12 |
| exp04 | focal | 3 | 0.8638 | 0.1096 | 0.8632 | 0.692 | 2.94 |

**Reading:** Cross-entropy gives the **best top-1 accuracy (87.90%)**. Focal loss does **not** beat it
on accuracy at any γ, and accuracy decreases monotonically as γ grows (87.44 → 87.28 → 86.38).
However, **focal loss at γ=1 more than halves the calibration error** (ECE 0.0154 vs CE 0.0337). At
γ≥2 calibration actually gets *worse* than CE for this small model — the strong down-weighting makes
it under-confident. So on balanced data the only FL benefit is calibration, and only at low γ.

## Long-tailed CIFAR-10-LT, imbalance factor 100 (exp05–exp07)
Train class histogram: [5000, 2997, 1797, 1077, 646, 387, 232, 139, 83, 50] (test stays balanced).
| exp | loss | γ | test acc | ECE | macro-F1 | worst-class acc | train min |
|-----|------|---|----------|-----|----------|-----------------|-----------|
| exp05 | CE | – | 0.5766 | 0.2202 | 0.5332 | 0.036 | 2.49 |
| exp06 | focal | 2 | 0.5908 | 0.0967 | 0.5618 | 0.114 | 1.77 |
| exp07 | focal | 3 | **0.5910** | **0.0503** | **0.5655** | **0.133** | 1.49 |

**Reading:** Under imbalance the verdict **flips**. Focal loss beats cross-entropy on **every** metric:
- top-1 accuracy +1.4pp (0.5766 → 0.5910),
- macro-F1 +3.2pp (0.5332 → 0.5655),
- **worst (rarest) class accuracy 0.036 → 0.133 — a 3.7× improvement**,
- ECE 0.2202 → 0.0503 — a **4.4× reduction** in calibration error.
Higher γ helps more here (γ=3 ≥ γ=2), the opposite of the balanced regime, because aggressive
down-weighting of the abundant easy head classes lets the rare tail classes contribute more gradient.

## Conclusion
**Answer to the research question: No — focal loss does not beat cross-entropy for CIFAR-10
classification with a small CNN on the standard balanced dataset.** Cross-entropy wins on accuracy
(87.90% vs FL's best 87.44%). Focal loss's value is conditional:
1. **Balanced data:** FL ties CE on accuracy at best and helps only *calibration*, and only at low
   γ (γ=1 halves ECE; γ≥2 hurts both accuracy and calibration).
2. **Imbalanced data:** FL clearly wins — better accuracy, macro-F1, calibration, and a large gain in
   rare-class accuracy, with larger γ preferred.

This exactly matches the literature (Lin 2017; Mukhoti 2020; Cui 2019): focal loss is a tool for
*imbalance and calibration*, not a drop-in accuracy upgrade on balanced benchmarks.

**Validity:** CE baseline reached 87.90% ≥ 75% target → comparison is trustworthy. Full 7-run matrix
completed in 1 iteration, no reruns needed. Stop condition met (matrix complete, baseline valid).
Total GPU time ≈ 20 minutes.

**Best configs:** balanced accuracy → CE (exp01); balanced calibration → focal γ=1 (exp02);
imbalanced (all metrics) → focal γ=3 (exp07).
