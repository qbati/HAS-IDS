# AOC-IDS Baseline

## Upstream Source

**Original Repository:** https://github.com/xinchen930/AOC-IDS

**Reference:**
```
Zhang, X., et al. (2024). AOC-IDS: Autonomous Online Framework with 
Contrastive Learning for Intrusion Detection. Future Generation Computer Systems.
```

## Our Modifications

This directory contains our wrapper/evaluation scripts for running AOC-IDS on UNSW-NB15 with:

- **Extended metrics:** ROC-AUC, PR-AUC, FPR, per-class recall
- **Computation cost tracking:** Training and testing time measurements
- **Result averaging:** Automated averaging across multiple seeds

### Files

- `run_aoc_ids_unsw.py` - Main script with 5-seed evaluation loop
- `utils_aoc.py` - Utility functions (adapted from upstream)
- `eval_classes.py` - Per-class recall evaluation
- `eval_classes_averager.py` - Averages results across seeds

## Usage

```bash
cd baselines/aocids
python run_aoc_ids_unsw.py
```

**Note:** The upstream AOC-IDS code must be obtained separately if you need the original implementation. Our scripts are modified versions tailored for our evaluation protocol.

## Outputs

Results are saved to `results_aoc_github/`:
- Per-seed predictions: `aoc_ids_github_unsw_seed{N}_predictions.csv`
- Per-seed metrics: `aoc_ids_github_unsw_seed{N}_metrics.json`
- Averaged metrics: `aoc_ids_github_unsw_metrics_AVERAGED.json`
- Averaged predictions: `aoc_ids_github_unsw_predictions_AVERAGED.csv`
