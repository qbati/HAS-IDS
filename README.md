# HAS-IDS: Hybrid Anomaly Scoring for Low-Frequency Network Intrusion Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18481511.svg)](https://doi.org/10.5281/zenodo.18481511)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

**Authors:** Salah Abdullah Khalil Abdulrahman¹, Vera Suryani¹, Grzegorz Kołaczek²  
**Affiliations:**  
¹ School of Computing, Telkom University, Bandung, West Java, Indonesia  
² Computer Science and Systems Engineering, Wrocław University of Science and Technology, Wrocław, Lower Silesia, Poland

---

## Overview

HAS-IDS addresses the critical challenge of detecting rare network attacks (<1% & <0.5% prevalence) in highly imbalanced intrusion detection datasets. The system combines symmetric supervised contrastive learning with multi-evidence anomaly scoring to achieve superior recall on low-frequency attack classes while maintaining practical false-positive rates.

**Core Innovation:** A meta-classifier fuses three complementary evidence streams BGMM-based probabilistic normality, ANN-based attack similarity, and mixture-level contextual features into a calibrated decision framework operating under fixed FPR constraints.

---

## Architecture

**Module 1: Representation Learning**
- Attention-guided encoder with symmetric supervised contrastive loss
- Preserves inter-attack topology while compacting normal traffic
- Fisher z-transform for variance-stabilized scoring

**Module 2: Hybrid Anomaly Scoring**
- Bayesian Gaussian Mixture Model for probabilistic normality estimation
- FAISS-accelerated approximate nearest neighbor search over attack bank
- Contextual mixture statistics (mean, variance of dominant component)
- Logistic meta-classifier with calibration-split threshold selection

**Module 3: Inference Pipeline**
- Fixed FPR budget enforcement (α-constrained decision thresholds)
- Online deployment with frozen encoder and cached artifacts

---

## Requirements

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Python 3.10
- PyTorch 2.5.1 (GPU-accelerated)
- scikit-learn 1.6.0
- faiss-cpu 1.7.4 (or faiss-gpu for CUDA acceleration)
- pandas, numpy

**Hardware:**
- Tested on Intel Core i7-12700H, 16GB RAM, NVIDIA RTX 3070 Ti (8GB VRAM), Windows 11

---

## Quick Start

```bash
# UNSW-NB15
cd HAS-IDS
python has_ids_unsw.py

# CIC-IDS2017
python has_ids_dcic.py
```

**Required Dataset Structure:**
```
Datasets/
├── UNSW/
│   ├── BUNSWTrain.csv, BUNSWTest.csv
│   └── MUNSWTrain.csv, MUNSWTest.csv
└── DCIC2017/
    ├── DBcic2017_train.csv, DBcic2017_test.csv
    └── DMcic2017_train.csv, DMcic2017_test.csv
```

**Note:** Datasets are not included in this repository. Download from:
- **UNSW-NB15:** [Official Source](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **CIC-IDS2017 (Corrected):** [DistriNet](https://intrusion-detection.distrinet-research.be/WTMC2021/) or [Kaggle](https://www.kaggle.com/datasets/dhoogla/distrinetcicids2017/data)

See [USAGE.md](USAGE.md) for dataset preparation and preprocessing instructions.

---

## Project Structure

```
HAS-IDS/
├── HAS-IDS/                     # Main implementation
│   ├── has_ids_unsw.py          # UNSW-NB15 experiments
│   ├── has_ids_dcic.py          # CIC-IDS2017 experiments
│   └── eval_classes.py          # Per-class evaluation
├── Ablation_Study/              # Component ablations (14 variants)
│   ├── full_has_ids_unsw.py
│   ├── full-has_ids_cicids2017.py
│   ├── has-ids_*_wo_prob.py
│   ├── has-ids_*_wo_contextualScores.py
│   ├── has-ids_*_wo_instanceScore.py
│   ├── has-ids_*_probOnly.py
│   ├── has-ids_*_contextOnly.py
│   └── has-ids_*_AnnOnly_woBgmm.py
├── baselines/                   # Comparative methods
│   ├── CL-BGMM/                 # Contrastive+BGMM baseline
│   │   ├── cl_bgmm_unsw.py
│   │   ├── cl_bgmm_dcic.py
│   │   └── eval_classes.py
│   ├── aocids/                  # AOC-IDS wrapper
│   │   ├── run_aoc_ids_unsw.py
│   │   ├── utils_aoc.py
│   │   ├── eval_classes.py
│   │   ├── eval_classes_averager.py
│   │   └── README.md
│   ├── CIDS/                    # CIDS implementation
│   │   └── cids_unsw.py
│   ├── autoencoder/             # Deep autoencoder baseline
│   │   ├── run_unsw_ae.py
│   │   └── run_dcic_ae.py
│   └── Isolation_forest/        # Isolation Forest baseline
│       ├── isolation_forest_unsw.py
│       └── isolation_forest_dcic.py
├── Dataset_split/               # Preprocessing utilities
│   ├── split.py
│   ├── DBsplit.py
│   ├── DMsplit.py
│   └── Dcicids2017.py
├── Dataset_difficulty/          # Dataset analysis scripts
│   ├── unsw.py
│   └── dcic2017.py
├── Visual_t-SNE_histo/          # Visualization tools
│   ├── has-ids_unsw_vis.py
│   └── has-ids_dcic_vis.py
├── requirements.txt
├── USAGE.md
├── CITATION.bib
└── LICENSE
```

---

## Reproducibility

All experiments use fixed random seeds (default: 42). Hyperparameters are embedded in source files. For detailed configuration and ablation variants, consult [USAGE.md](USAGE.md).

**Key Hyperparameters:**
- Encoder: 196→128→64→32 (UNSW), 82→256→128→128 (CIC-IDS2017)
- Contrastive temperature: τ=0.38 (UNSW), τ=0.05 (CIC-IDS2017)
- BGMM components: K=5, reg=1e-2 (UNSW); K=3, reg=4e-3 (CIC-IDS2017)
- ANN neighbors: k=7 (all datasets)
- FPR budget: α=0.10 (all datasets)

---

## Baselines

Comparative implementations provided in `baselines/`:
- **CL-BGMM:** Contrastive learning + BGMM (primary baseline)
- **AOC-IDS:** Online contrastive IDS with drift adaptation
- **CIDS:** Contrastive-enhanced supervised IDS
- **Isolation Forest:** Tree-based anomaly detector
- **Autoencoder:** Reconstruction-error-based detector

---

## Citation

```bibtex
@software{abdulrahman2026hasids_code,
  author    = {Abdulrahman, Salah Abdullah Khalil and Suryani, Vera and 
               Ko{\l}aczek, Grzegorz},
  title     = {HAS-IDS: Hybrid Anomaly Scoring for Low-Frequency 
               Network Intrusion Detection},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/qbati/HAS-IDS}
}
```

For the research paper, see [CITATION.bib](CITATION.bib) for publication-ready BibTeX entries.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

**Salah Abdullah Khalil Abdulrahman**  
School of Computing, Telkom University  
Email: salahabdullahkhalil@student.telkomuniversity.ac.id
