# Research Status: Focal Loss vs Cross-Entropy for CIFAR-10 (Small CNN)

- **Slug**: focal-loss-cifar10
- **Statement**: Does focal loss beat cross-entropy for CIFAR-10 classification with a small CNN?
- **Started**: 2026-06-04
- **Author block**: Default Vizuara block (Naman Dwivedi, Raj Dandekar, Rajat Dandekar, Sreedath Panat — Vizuara AI Labs)

## Pipeline Stages
| # | Stage | Status | Artifact | Notes |
|---|-------|--------|----------|-------|
| 1 | Literature review     | ✅ done | literature/literature_review.md | 3 papers downloaded; clear hypothesis |
| 2 | Experiment planning   | ✅ done | plan/experiment_plan.md | 8-run matrix |
| 3 | Experiments (Modal)   | ✅ done | experiments/experiments_log.md | 7 runs + smoke; ~20 min A100 |
| 4 | Paper outline         | ✅ done | paper/outline.md | |
| 5 | Figures               | ✅ done | paper/figures/output/ | 6 figs, all verified |
| 6 | Write paper           | ✅ done | paper/paper.tex | |
| 7 | Compile paper         | ✅ done | paper/paper.pdf | 4 pages, refs resolved |
| 8 | GitHub repo           | ✅ done | https://github.com/OmuNaman/focal-loss-cifar10 | |
| 9 | Project website       | ✅ done | https://omunaman.github.io/focal-loss-cifar10/ | Live (removed dead itsnaman.me CNAME from user-site repo) |

Repo: https://github.com/OmuNaman/focal-loss-cifar10
Website: http://itsnaman.me/focal-loss-cifar10/

## Key Decisions Log
- 2026-06-04: Slug = `focal-loss-cifar10`. Dry-run validation of the full pipeline on a small, cheap, well-understood problem.

## Current Best Result
**Answer: No — CE beats focal loss on balanced CIFAR-10 accuracy (87.90% vs 87.44%).**
- Balanced accuracy: CE 87.90% (exp01) > focal best 87.44% (exp02, γ=1)
- Balanced calibration: focal γ=1 ECE 0.0154 < CE 0.0337 (focal wins calibration)
- Imbalanced (LT-100): focal γ=3 beats CE on all metrics — acc 0.591 vs 0.577, worst-class 0.133 vs 0.036 (3.7×), ECE 0.050 vs 0.220 (4.4×)
- Validity: CE baseline 87.90% ≥ 75% target ✅
