# Figure Generation Report

## Summary
- Total: 6 figures (2 diagrams via PaperBanana + 4 data plots via matplotlib)
- PASSED first try: 6
- PASSED after fixes: 0
- FAILED: 0

## Status
| # | id | type | status | attempts | notes |
|---|----|------|--------|----------|-------|
| 1 | fig1_design | diagram | PASS | 1 | Design overview; all labels correct, pastel, white bg |
| 2 | fig2_architecture | diagram | PASS | 1 | CNN architecture; channel counts correct |
| 3 | fig3_balanced | plot | PASS | 1 | Dual-axis acc vs ECE; exact numbers from exp01-04 |
| 4 | fig4_imbalanced | plot | PASS | 1 | CE vs focal grouped bars; exact from exp05-07 |
| 5 | fig5_perclass | plot | PASS | 1 | Per-class accuracy vs frequency; tail rescue clear |
| 6 | fig6_curves | plot | PASS | 1 | Test acc per epoch from curve.csv |

Notes:
- Data plots generated with matplotlib from results.jsonl / curve.csv (exact, reproducible) via
  `figures/make_plots.py`.
- Diagrams generated with PaperBanana (gemini-3-pro-image). The known critic-parse warning
  (Bug 5) appeared on fig1 but the image saved correctly (final_output.png), so no impact.
