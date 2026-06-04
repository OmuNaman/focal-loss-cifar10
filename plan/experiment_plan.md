# Experiment Plan: Focal Loss vs Cross-Entropy for CIFAR-10 (Small CNN)

_Generated 2026-06-04_

## Hypothesis
Grounded in the literature review, we predict three outcomes:
1. **Balanced CIFAR-10:** focal loss (FL) will **not** beat cross-entropy (CE) on top-1
   accuracy (it will match or slightly trail CE), because the balanced 10-class setting lacks
   the easy-negative dominance FL was designed to suppress.
2. **Calibration:** FL will achieve **lower ECE** (better calibration) than CE at comparable
   accuracy (Mukhoti et al. 2020).
3. **Imbalanced CIFAR-10-LT (factor 100):** FL will **beat plain CE** on overall and worst-class
   accuracy, because down-weighting easy majority examples helps under imbalance.

Confirming/refuting (1) directly answers the research question; (2) and (3) characterise *when*
FL is worth using.

## Approach
Single, fixed small CNN architecture trained identically except for the loss function and the
data distribution. We vary only: loss in {CE, FL(gamma)} and split in {balanced, LT-100}.
This isolates the loss effect. FL with gamma=0 reduces to CE, giving a clean control.

**Model (small CNN, ~1.2M params):** 3 conv blocks (32->64->128 channels), each
Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU -> MaxPool, then global average
pool -> FC(128) -> Dropout(0.3) -> FC(10). Optimizer AdamW (lr 1e-3, wd 5e-4), cosine schedule,
batch size 128, standard CIFAR augmentation (random crop + horizontal flip + normalization).

## Datasets
| Dataset | Download method | Size | Preprocessing | Split |
|---------|-----------------|------|---------------|-------|
| CIFAR-10 (balanced) | `torchvision.datasets.CIFAR10(download=True)` into the Modal Volume | 50k train / 10k test | per-channel normalize; train aug: RandomCrop(32,pad=4)+HFlip | official train/test |
| CIFAR-10-LT (factor 100) | derived by exponential subsampling of the CIFAR-10 train set inside the training fn | ~12.4k train / 10k test | same normalize/aug; test stays balanced | seed-fixed subsample |

## Baselines
| Baseline | Source | Reported metric | We reproduce? |
|----------|--------|-----------------|---------------|
| CE small CNN, balanced | general literature | ~75-85% top-1 | Yes — this is exp01, our anchor |
| FL ~= CE on balanced acc | Mukhoti 2020 / TDS | acc parity, lower ECE | Yes — exp02-04 |
| FL > CE on long-tailed | Cui 2019 | acc gain under imbalance | Yes — exp05-07 |

## Success Criterion
- **Primary metric:** test top-1 accuracy. The scientific answer is the **sign of (FL − CE)**
  on each split.
- **Validity anchor (the "target"):** the CE baseline (exp01) must train to **>= 75% test
  top-1 accuracy** on balanced CIFAR-10. If the baseline is sane, the comparison is trustworthy
  and the study is "satisfactory" once the full matrix has run.
- **Secondary metrics:** Expected Calibration Error (ECE, 15 bins), macro-F1, worst-class
  accuracy (for the imbalanced split), final train/val loss.

## Stop Conditions
- Stop when the **full experiment matrix below has completed** with a valid CE baseline
  (>=75% acc), OR
- Stop after **12** experiment iterations (max-iteration cap), reporting the best/most complete
  results obtained. (Matrix is 8 runs, leaving headroom for up to 4 reruns/fixes.)

## Improvement Rule (what to try, in order)
1. If exp01 (CE baseline) < 75% acc: increase epochs 30 -> 50, then check augmentation/lr.
2. If any run diverges (NaN/flat): lower lr to 5e-4, verify FL numerics (clamp p_t).
3. If FL underperforms wildly on balanced (>3pp below CE): sweep gamma down (2 -> 1).
4. Once the matrix is complete and the baseline is valid: stop. Do not chase further gains —
   the question is comparative, not a leaderboard.

## Experiment Matrix
| exp_id | type | loss | split | gamma | epochs | GPU | est. runtime |
|--------|------|------|-------|-------|--------|-----|--------------|
| exp00 | smoke | CE | balanced (2k subset) | – | 1 | A100 | <1 min |
| exp01 | baseline | CE | balanced | – | 30 | A100 | ~3 min |
| exp02 | proposed | FL | balanced | 1.0 | 30 | A100 | ~3 min |
| exp03 | proposed | FL | balanced | 2.0 | 30 | A100 | ~3 min |
| exp04 | proposed | FL | balanced | 3.0 | 30 | A100 | ~3 min |
| exp05 | baseline | CE | LT-100 | – | 30 | A100 | ~2 min |
| exp06 | proposed | FL | LT-100 | 2.0 | 30 | A100 | ~2 min |
| exp07 | proposed | FL | LT-100 | 3.0 | 30 | A100 | ~2 min |

All runs use GPU **A100-40GB** (default). The small CNN fits easily in <2GB; 80GB is not needed.

## Risks
- **Baseline too weak** (small CNN underfits): mitigated by 30 epochs + cosine LR + augmentation;
  improvement rule raises epochs if <75%.
- **FL numerical instability** (log of small p): clamp probabilities to [1e-7, 1] in the FL impl.
- **CIFAR-10-LT construction bug**: fix the random seed and assert the per-class counts follow
  the intended exponential profile before training; log the class histogram.
- **Total cost:** 8 short runs on A100 ≈ ~20 min GPU ≈ well under $2 of the $30 credit.
