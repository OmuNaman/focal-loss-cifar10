# Paper Outline: When Does Focal Loss Help? A Controlled Study on CIFAR-10 with a Small CNN

## Metadata
- **Topic**: Focal loss vs cross-entropy for small-CNN image classification
- **Status**: 🟢 Strong, complete results (7 runs + smoke, valid baseline)
- **Paper Type**: Full empirical study
- **Author block**: Vizuara default

## Title
"When Does Focal Loss Help? A Controlled Comparison of Focal Loss and Cross-Entropy for
Small-CNN Image Classification on Balanced and Long-Tailed CIFAR-10"

## Authors
Naman Dwivedi¹, Raj Dandekar¹, Rajat Dandekar¹, Sreedath Panat¹
¹Vizuara AI Labs, hello@vizuara.com

## Abstract (~200 words)
Focal loss was introduced to address foreground-background imbalance in dense object detection,
yet it is often adopted as a generic drop-in replacement for cross-entropy in image
classification. We ask a precise question: does focal loss beat cross-entropy for CIFAR-10
classification with a small CNN? Holding a 0.31M-parameter CNN, optimizer, schedule, and
augmentation fixed, we vary only the loss (cross-entropy vs focal loss with γ∈{1,2,3}) and the
class distribution (balanced CIFAR-10 vs long-tailed CIFAR-10-LT, imbalance factor 100), and
report top-1 accuracy, Expected Calibration Error (ECE), macro-F1, and worst-class accuracy.
On balanced CIFAR-10, cross-entropy attains the best accuracy (87.90%); focal loss never
exceeds it and degrades monotonically with γ, its only benefit being improved calibration at
γ=1 (ECE 0.0154 vs 0.0337). Under class imbalance the conclusion reverses: focal loss (γ=3)
improves accuracy (59.1% vs 57.7%), macro-F1 (+3.2 points), worst-class accuracy (3.6%→13.3%,
a 3.7× gain), and calibration (ECE 0.220→0.050). We conclude focal loss is a tool for imbalance
and calibration, not a generic accuracy upgrade, and give practical guidance on the γ setting.

## I. Introduction
- Para 1: Loss functions for classification; focal loss's origin in detection (cite Lin 2017).
- Para 2: Its frequent uncritical reuse as a CE replacement; the open practical question.
- Para 3: Our controlled design — vary only loss and class distribution.
- Para 4: Contributions (numbered):
  1. A clean, fully-reproducible head-to-head of CE vs focal loss (γ∈{1,2,3}) on balanced and
     long-tailed CIFAR-10 with a fixed small CNN.
  2. Evidence that on balanced data CE wins accuracy and focal helps only calibration (low γ).
  3. Evidence that under imbalance focal reverses the verdict on all metrics, with large γ best.
  4. Practical guidance + open code/results on Modal.
- Para 5: Paper organization.

## II. Background and Related Work
### A. Focal Loss and Class Imbalance — Lin 2017; Cui 2019 (CB), Cao 2019 (LDAM)
### B. Calibration of Neural Networks — Guo 2017; Mukhoti 2020
### C. Long-Tailed Recognition Benchmarks — Cui 2019; Buda 2018

## III. Methodology
### A. Problem Formulation
- Softmax CE: Eq. (1). Focal loss: FL(p_t) = -(1-p_t)^γ log(p_t), Eq. (2). γ=0 ⇒ CE.
### B. Small CNN Architecture
- 3 conv blocks (32-64-128), BN+ReLU, global avg pool, FC head, dropout 0.3. 0.31M params.
- Hyperparameter table (Table I).
### C. Long-Tailed Data Construction
- Exponential subsampling, imbalance factor 100; class histogram; balanced test set. Eq. (3).
### D. Evaluation Metrics
- Top-1 acc, ECE (15-bin, Eq. 4), macro-F1, worst-class accuracy.

## IV. Experimental Setup
### A. Dataset (CIFAR-10; CIFAR-10-LT factor 100 — Table of stats)
### B. Implementation Details (Modal serverless A100-40GB; PyTorch 2.4; AdamW lr 1e-3, wd 5e-4;
     cosine; 30 epochs; batch 128; RandomCrop+HFlip; fixed seed)
### C. Compared Configurations (CE; focal γ=1,2,3; two splits)

## V. Results
### A. Balanced CIFAR-10 (Table II) — CE best accuracy; focal γ=1 best ECE; γ↑ hurts.
   - Fig: accuracy + ECE by configuration.
### B. Long-Tailed CIFAR-10-LT (Table III) — focal wins all metrics; γ=3 best.
   - Fig: CE vs focal accuracy/macro-F1/worst-class.
   - Fig: per-class accuracy vs class frequency (the tail-rescue figure).
### C. Discussion — why the verdict flips; the γ trade-off (calibration vs under-confidence).

## VI. Limitations and Future Work
- Single architecture/dataset; no temperature scaling; one imbalance factor; one seed per cell.
- Future: multiple seeds + error bars, ResNet, ImageNet-LT, focal+reweighting (CB-Focal), TS.

## VII. Conclusion
- Focal loss is conditional: imbalance/calibration tool, not a generic accuracy upgrade.

## References
(12-18 from literature_review.md §8; all real)

---

## Figure Master List

### Figure 1: Experimental Design Overview — diagram (`generate`) — HIGH (hero)
- Content: research question → fixed small CNN → 2 losses (CE, focal) × 2 splits (balanced,
  LT-100) → 4 metrics (acc, ECE, macro-F1, worst-class). Boxes + arrows, pastel.
- Caption: "Controlled experimental design: only the loss function and the class distribution
  are varied; architecture, optimizer, schedule, and augmentation are held fixed."

### Figure 2: Small CNN Architecture — diagram (`generate`) — HIGH
- Content: input 32×32×3 → 3 conv blocks (32→64→128, each ConvBN-ReLU×2 + MaxPool) → GAP →
  FC128 → Dropout → FC10. Label channel sizes.
- Caption: "The fixed 0.31M-parameter convolutional network used in all experiments."

### Figure 3: Balanced CIFAR-10 — Accuracy vs Calibration — plot (`plot`) — HIGH
- Data: `fig3_balanced.csv` from exp01-04 (config, test_acc, ece).
- Content: grouped bars: test accuracy and ECE per config (CE, FL γ=1/2/3).
- Caption: "On balanced CIFAR-10, cross-entropy yields the best accuracy while focal loss (γ=1)
  yields the best calibration; larger γ degrades both."

### Figure 4: Long-Tailed CIFAR-10-LT — CE vs Focal — plot (`plot`) — HIGH
- Data: `fig4_imbalanced.csv` from exp05-07 (config, test_acc, macro_f1, worst_class_acc).
- Content: grouped bars of accuracy / macro-F1 / worst-class accuracy for CE vs FL γ=2 / γ=3.
- Caption: "Under imbalance (factor 100), focal loss improves accuracy, macro-F1, and especially
  worst-class accuracy over cross-entropy."

### Figure 5: Per-Class Accuracy vs Class Frequency (LT-100) — plot (`plot`) — HIGH (money figure)
- Data: `fig5_perclass.csv` — per-class accuracy for exp05 (CE) and exp07 (FL γ=3), classes
  ordered by training frequency (5000→50).
- Content: line/bar across 10 classes; CE vs FL γ=3; show tail-class rescue.
- Caption: "Per-class accuracy ordered by training frequency. Focal loss recovers accuracy on
  the rarest classes (rightmost) where cross-entropy collapses."

### Figure 6: Test Accuracy vs Epoch — plot (`plot`) — MEDIUM
- Data: curve.csv for exp01/exp03/exp05/exp07.
- Content: test accuracy across epochs for representative configs.
- Caption: "Training dynamics: test accuracy per epoch for representative configurations."

## Tables Plan
- Table I: Architecture / training hyperparameters.
- Table II: Balanced CIFAR-10 results (exp01-04): acc, ECE, macro-F1, worst-class.
- Table III: Long-tailed CIFAR-10-LT results (exp05-07): acc, ECE, macro-F1, worst-class.

## Equations Plan
- Eq. 1: softmax cross-entropy. Eq. 2: focal loss. Eq. 3: exponential LT class counts.
- Eq. 4: Expected Calibration Error.

## Placeholder Sections
- None — all results are real and complete.
