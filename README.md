# HAS-IDS: Hybrid Anomaly Scoring for Low-Frequency Network Intrusion Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20032288.svg)](https://doi.org/10.5281/zenodo.20032288)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

**Authors:** Salah Abdullah Khalil Abdulrahman¹, Vera Suryani¹, Grzegorz Kołaczek²  
**Affiliations:**  
¹ School of Computing, Telkom University, Bandung, West Java, Indonesia  
² Computer Science and Systems Engineering, Wrocław University of Science and Technology, Wrocław, Lower Silesia, Poland

---

## Overview

HAS-IDS addresses the challenge of detecting rare network attacks in highly imbalanced intrusion detection datasets. The system combines supervised contrastive representation learning with multi-evidence anomaly scoring to improve recall on low-frequency attack classes while maintaining practical false-positive rates.

**Core contribution:** HAS-IDS fuses three complementary evidence streams: BGMM-based probabilistic normality, ANN-based attack similarity, and mixture-level contextual features. These streams are combined through a calibrated meta-classifier operating under a fixed false-positive-rate budget.

---

## Architecture

**Module 1: Representation Learning**
- Attention-guided encoder with supervised contrastive loss.
- Fisher z-transform for variance-stabilized scoring.

**Module 2: Hybrid Anomaly Scoring**
- Bayesian Gaussian Mixture Model for probabilistic normality estimation.
- FAISS-accelerated nearest-neighbour search over the attack bank.
- Contextual mixture statistics from the dominant BGMM component.
- Logistic meta-classifier with calibration-split threshold selection.

**Module 3: Inference Pipeline**
- FPR-constrained decision thresholding.
- High-confidence-normal guard.
- Online inference with frozen encoder and cached artifacts.

---

## Requirements

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Core dependencies:

```text
Python 3.10
PyTorch 2.5.1
scikit-learn 1.6.0
faiss-cpu 1.7.4
pandas
numpy
```

FAISS is recommended. If FAISS is unavailable, scripts that include a fallback path use NumPy-based similarity search where implemented.

Reported experiments were run on:

```text
Intel Core i7-12700H
16 GB RAM
NVIDIA RTX 3070 Ti, 8 GB VRAM
```

---

## Dataset Availability

Datasets are not included in this repository. Download them from the original public sources.

### UNSW-NB15

Official source:

```text
https://research.unsw.edu.au/projects/unsw-nb15-dataset
```

Expected directory structure:

```text
Datasets/UNSW/
├── BUNSWTrain.csv
├── BUNSWTest.csv
├── MUNSWTrain.csv
└── MUNSWTest.csv
```

### Corrected Distrinet-CIC-IDS2017

Recommended source:

```text
https://intrusion-detection.distrinet-research.be/WTMC2021/
```

Kaggle mirror:

```text
https://www.kaggle.com/datasets/dhoogla/distrinetcicids2017/data
```

Expected directory structure:

```text
Datasets/DCIC2017/
├── DBcic2017_train.csv
├── DBcic2017_test.csv
├── DMcic2017_train.csv
└── DMcic2017_test.csv
```

The manuscript uses fixed stratified train/test partitions with:

```text
random_state = 42
```

See [USAGE.md](USAGE.md) for full dataset preparation and execution instructions.

---

## Quick Start

From the repository root:

```bash
cd HAS-IDS
```

Run UNSW-NB15:

```bash
python has_ids_unsw.py
```

Run corrected Distrinet-CIC-IDS2017:

```bash
python has_ids_dcic.py
```

Main outputs include prediction CSV files, metrics JSON files, and saved model artifacts.

---

## Reproducibility: 10-Run Evaluation

The revised manuscript reports mean ± standard deviation for HAS-IDS over 10 independent runs.

Run the scripts from inside the `HAS-IDS/` directory:

```bash
cd HAS-IDS
```

### UNSW-NB15

```bash
python run_10seeds_unsw.py
```

This reproduces the HAS-IDS mean ± std values for Tables 6 and 8.

Seed list:

```text
42, 0, 1, 2, 3, 4, 5, 6, 7, 8
```

Outputs:

```text
unsw_10seeds_raw_results.csv
unsw_10seeds_summary.txt
```

### Corrected Distrinet-CIC-IDS2017

```bash
python run_10seeds_dcic.py
```

This reproduces the HAS-IDS mean ± std values for Tables 10 and 12.

Seed list:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

Outputs:

```text
dcic_10seeds_raw_results.csv
dcic_10seeds_summary.txt
```

Before running either script, check the `CONFIGURE ONLY THESE` block at the top of the script and update dataset paths if needed.

---

## Project Structure

```text
HAS-IDS/
├── HAS-IDS/                     # Main implementation
│   ├── has_ids_unsw.py          # UNSW-NB15 single-run experiment
│   ├── has_ids_dcic.py          # Corrected Distrinet-CIC-IDS2017 single-run experiment
│   ├── run_10seeds_unsw.py      # 10-run reproducibility script for UNSW-NB15
│   ├── run_10seeds_dcic.py      # 10-run reproducibility script for corrected Distrinet-CIC-IDS2017
│   └── eval_classes.py          # Per-class evaluation utility
├── Ablation_Study/              # Evidence-stream ablation scripts
├── baselines/                   # Baseline implementations
├── Dataset_split/               # Dataset preprocessing and split utilities
├── Dataset_difficulty/          # Dataset analysis scripts
├── Visual_t-SNE_histo/          # Visualization scripts
├── requirements.txt
├── USAGE.md
├── CITATION.bib
└── LICENSE
```

---

## Key Hyperparameters

| Hyperparameter | UNSW-NB15 | Corrected Distrinet-CIC-IDS2017 |
|---|---:|---:|
| Encoder | 196-128-64-32 | 82-256-128-128 |
| Feature dimension | 32 | 128 |
| SCL temperature | 0.38 | 0.05 |
| Epochs | 18 | 50 |
| Batch size | 128 | 256 |
| Learning rate | 0.0018 | 0.0010 |
| Optimizer | Adam | Adam |
| BGMM setting | K = 5, reg = 1e-2 | K = 3, reg = 4e-3 |
| FAISS neighbours | 7 | 7 |
| Guard quantile | 0.995 | 0.995 |
| Target FPR | 0.10 | 0.10 |

---

## Threshold Search Note

In `v1.0.3`, the corrected Distrinet-CIC-IDS2017 implementation evaluates threshold candidates using the observed calibration scores directly, rather than a rounded score grid.

This avoids rounding-induced degenerate all-Normal predictions in saturated multi-seed runs while keeping the FPR constraint and tie-breaking logic unchanged.

---

## Ablation Study

Ablation scripts are stored in:

```text
Ablation_Study/
```

These scripts evaluate the contribution of individual evidence streams, including:

```text
Full HAS-IDS
without BGMM/probabilistic stream
without contextual stream
without instance/ANN stream
probability-only
context-only
ANN-only
```

For corrected Distrinet-CIC-IDS2017, the main evaluation uses learning rate `η = 0.0010`, while the ablation study uses a diagnostic setting `η = 0.0020` to keep stream-contribution differences visible. The main benchmark results remain those reported by `has_ids_dcic.py` and `run_10seeds_dcic.py`.

---

## Baselines

Baseline implementations are stored in:

```text
baselines/
```

Included baselines:

```text
CL-BGMM
AOC-IDS
CIDS
Autoencoder
Isolation Forest
```

Some baselines are dataset-specific. Check the script names and folder-level comments before execution.

---

## Changelog

### v1.0.3

- Updated corrected Distrinet-CIC-IDS2017 threshold search to use observed calibration scores directly instead of a rounded score grid.
- Added 10-run reproducibility scripts:
  - `run_10seeds_unsw.py`
  - `run_10seeds_dcic.py`
- Updated usage documentation and reproducibility instructions.
- No dataset split, model architecture, baseline implementation, or experimental protocol was changed.

### v1.0.2

- Updated repository documentation and metadata for revised manuscript preparation.

### v1.0.1

- Initial public release.

---

## Citation

Use [CITATION.bib](CITATION.bib) for citation metadata.

Temporary pre-Zenodo `v1.0.3` citation:

```bibtex
@software{abdulrahman2026hasids_code,
  author    = {Abdulrahman, Salah Abdullah Khalil and Suryani, Vera and
               Ko{\l}aczek, Grzegorz},
  title     = {{HAS-IDS}: Hybrid Anomaly Scoring for Low-Frequency
               Network Intrusion Detection},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.3},
  doi       = {10.5281/zenodo.20032288},
  url       = {https://doi.org/10.5281/zenodo.20032288}
}
```

After the Zenodo `v1.0.3` archive is published, update this section with the final DOI.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Salah Abdullah Khalil Abdulrahman**  
School of Computing, Telkom University  
Email: salahabdullahkhalil@student.telkomuniversity.ac.id
