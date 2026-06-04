# Literature Review: Focal Loss vs Cross-Entropy for CIFAR-10 (Small CNN)

_Generated 2026-06-04 for: Does focal loss beat cross-entropy for CIFAR-10 classification with a small CNN?_

## 1. Summary
Focal loss (FL) was introduced by Lin et al. (2017) to combat the extreme foreground-background
class imbalance in one-stage dense object detection, by down-weighting the loss on easy,
well-classified examples so training focuses on hard examples. A consistent finding across the
literature is that on **balanced** classification benchmarks such as standard CIFAR-10, focal
loss does **not** improve top-1 accuracy over softmax cross-entropy (CE) and often slightly
trails it, because there are no easy-negative classes to suppress. Its documented benefits on
balanced data are in **calibration**: Mukhoti et al. (2020) show FL yields substantially lower
expected calibration error (ECE) than CE while preserving accuracy. On **imbalanced / long-tailed**
CIFAR-10, FL helps over plain CE but is itself outperformed by methods explicitly designed for
imbalance (Class-Balanced loss, LDAM). The gap our small study targets: a clean, controlled,
reproducible head-to-head of FL vs CE for a *small* CNN on *both* balanced and an artificially
imbalanced CIFAR-10, reporting accuracy and calibration.

## 2. Key Papers
| # | Title | Year | Venue | Method | Datasets | Headline result | Link | Local PDF |
|---|-------|------|-------|--------|----------|-----------------|------|-----------|
| 1 | Focal Loss for Dense Object Detection | 2017 | ICCV | Focal loss + RetinaNet | COCO | FL enables 1-stage detector to beat 2-stage SOTA | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | lin2017_focal_loss.pdf |
| 2 | Calibrating Deep Neural Networks using Focal Loss | 2020 | NeurIPS | FL for classification | CIFAR-10/100, etc. | FL gives well-calibrated models at equal accuracy | [arXiv:2002.09437](https://arxiv.org/abs/2002.09437) | mukhoti2020_focal_calibration.pdf |
| 3 | Class-Balanced Loss Based on Effective Number of Samples | 2019 | CVPR | CB re-weighting (incl. CB-Focal) | Long-tailed CIFAR | CB term beats plain FL/CE on imbalance | [arXiv:1901.05555](https://arxiv.org/abs/1901.05555) | cui2019_class_balanced_loss.pdf |
| 4 | Cyclical Focal Loss | 2022 | arXiv | FL with cyclical gamma schedule | CIFAR, ImageNet | Improves on FL/CE on balanced + imbalanced | [arXiv:2202.08978](https://arxiv.org/pdf/2202.08978) | (not downloaded) |
| 5 | LDAM-DRW (margin loss + deferred reweighting) | 2019 | NeurIPS | Label-distribution-aware margin | Long-tailed CIFAR | Strong long-tailed baseline, beats FL | [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) | (not downloaded) |

### Detailed notes
**Lin et al. 2017 (Focal Loss).** Defines FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t). The
modulating factor (1 - p_t)^gamma shrinks the loss contribution of easy examples (p_t -> 1).
gamma=2, alpha=0.25 are the canonical detection settings. The motivation is a ~1:1000
foreground-background imbalance; the mechanism is generic but its payoff scales with how much
"easy negative" mass dominates the gradient. Takeaway: on a balanced 10-class problem the
imbalance FL corrects for is largely absent.

**Mukhoti et al. 2020 (Calibration).** On classification (incl. CIFAR-10/100 with ResNet/Wide-ResNet),
FL trained models are markedly better calibrated (lower ECE) than CE, *without* sacrificing test
accuracy; with temperature scaling FL gives SOTA calibration. They give a principled, sample-
dependent rule for gamma (gamma=3 or a schedule gamma=5 for p<0.2 else 3). Takeaway: the right
metric to detect an FL benefit on balanced CIFAR-10 is calibration (ECE), not just accuracy.

**Cui et al. 2019 (Class-Balanced).** Introduces the "effective number of samples" reweighting,
applied on top of CE, FL, or sigmoid. On long-tailed CIFAR, CB-Focal beats plain FL and CE.
Takeaway: under imbalance, FL alone is a weak baseline; reweighting matters more than the
modulating factor.

## 3. Methods Landscape
- **Softmax Cross-Entropy (CE):** the default; strong accuracy on balanced data, tends to be
  over-confident (poor calibration).
- **Focal Loss (FL):** CE with a (1 - p_t)^gamma down-weighting of easy examples. Helps when
  easy examples dominate the gradient (imbalance/detection); on balanced data its main effect
  is better calibration and sometimes slightly lower accuracy.
- **Class-balancing / margin methods (CB loss, LDAM):** explicitly target class frequency; the
  right tool for long-tailed data, generally superior to FL there.
- **Calibration post-hoc (temperature scaling):** orthogonal; both CE and FL benefit, FL+TS is
  SOTA-calibrated.

## 4. Datasets (candidates)
| Dataset | Task | Size | Source / URL | License | Notes |
|---------|------|------|--------------|---------|-------|
| CIFAR-10 (balanced) | 10-class image cls | 50k train / 10k test, 32x32 RGB | torchvision.datasets.CIFAR10 | MIT-ish (research) | The primary benchmark in the statement |
| CIFAR-10-LT (long-tailed) | imbalanced cls | exponential decay subsample of CIFAR-10 | derived from CIFAR-10 (imbalance factor e.g. 100) | — | Standard imbalance probe; built by subsampling |

## 5. Baselines & SOTA
| Method | Dataset | Metric | Reported value | Source |
|--------|---------|--------|----------------|--------|
| CE (small/medium CNN) | balanced CIFAR-10 | top-1 acc | typically 70-85% small CNN, ~93%+ ResNet | general literature |
| FL | balanced CIFAR-10 | top-1 acc | ~= or slightly below CE | TDS / Mukhoti 2020 |
| FL | balanced CIFAR-10 | ECE | lower (better) than CE | Mukhoti 2020 |
| CE vs FL vs CB-Focal | long-tailed CIFAR-10 | top-1 acc | CB-Focal > FL > CE | Cui 2019 |

## 6. Standard Metrics
- **Top-1 accuracy** (primary question metric).
- **Per-class accuracy / macro-F1** (matters under imbalance).
- **Expected Calibration Error (ECE)** and reliability diagrams (where FL is expected to win).
- **Train/val loss curves** (training dynamics).

## 7. Gap & Opportunity
The literature strongly predicts: **on balanced CIFAR-10, FL will not beat CE on accuracy**, but
**may match it while improving calibration**, and **under induced imbalance FL should beat plain CE**.
No widely-cited paper runs this exact clean ablation for a *small* CNN with both balanced and
imbalanced splits and both accuracy + ECE reported together. That is precisely a tractable,
falsifiable study. Directions worth trying:
1. CE vs FL (gamma in {1,2,3}) on **balanced** CIFAR-10 — measure acc + ECE.
2. CE vs FL on **CIFAR-10-LT** (imbalance factor 100) — measure overall + worst-class acc.
3. Sweep gamma to characterise the accuracy/calibration trade-off.

## 8. Candidate References (for the paper)
1. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal Loss for Dense Object Detection," ICCV, 2017.
2. J. Mukhoti, V. Kulharia, A. Sanyal, S. Golodetz, P. Torr, P. Dokania, "Calibrating Deep Neural Networks using Focal Loss," NeurIPS, 2020.
3. Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples," CVPR, 2019.
4. K. Cao, C. Wei, A. Gaidon, N. Arechiga, T. Ma, "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS, 2019.
5. L. N. Smith, "Cyclical Focal Loss," arXiv:2202.08978, 2022.
6. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, 2017.
7. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. report (CIFAR-10), 2009.
8. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.
9. I. Loshchilov, F. Hutter, "SGDR / Decoupled Weight Decay (AdamW)," ICLR, 2017/2019.
10. S. Ioffe, C. Szegedy, "Batch Normalization," ICML, 2015.
11. N. Srivastava et al., "Dropout," JMLR, 2014.
12. M. Buda, A. Maki, M. Mazurowski, "A systematic study of the class imbalance problem in CNNs," Neural Networks, 2018.
