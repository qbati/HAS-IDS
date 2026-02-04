# HAS-IDS Usage Guide

Concise instructions for executing experiments and reproducing results.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Main Experiments](#main-experiments)
3. [Baseline Methods](#baseline-methods)
4. [Ablation Studies](#ablation-studies)
5. [Hyperparameter Configuration](#hyperparameter-configuration)
6. [Custom Dataset Evaluation](#custom-dataset-evaluation)

---

## Quick Start

```bash
# Activate environment
conda activate hasids

# Verify installation
python -c "import torch, faiss, sklearn, pandas; print('Ready')"

# Run first experiment
cd HAS_IDS
python has_ids_unsw.py
```

**Dataset Structure:**
```
Datasets/
├── UNSW/
│   ├── BUNSWTrain.csv, BUNSWTest.csv      # Binary labels
│   └── MUNSWTrain.csv, MUNSWTest.csv      # Multiclass labels
└── DCIC2017/
    ├── DBcic2017_train.csv, DBcic2017_test.csv
    └── DMcic2017_train.csv, DMcic2017_test.csv
```

---

## Main Experiments

**UNSW-NB15:**
```bash
cd HAS_IDS && python has_ids_unsw.py
```

**CIC-IDS2017:**
```bash
cd HAS_IDS && python has_ids_ids2017.py
```

Output: Model checkpoint (.joblib), predictions (.csv), and metrics (.json) files.

---

## Baseline Methods

| Method              | Command                                                 |
|---------------------|---------------------------------------------------------|
| **CL-BGMM**         | `cd CL-BGMM && python bgmm_unsw_bv2_fixedv2.py`        |
| **AOC-IDS**         | `cd baselines/aocids && python run_aoc_ids_unsw.py`    |
| **CIDS**            | `cd CIDS && python cids_unsw.py`                       |
| **Isolation Forest**| `cd Isolation_forest && python isolation_forest_unsw.py`|
| **Autoencoder**     | `cd autoencoder && python run_unsw_ae.py`              |
| **OneR**            | `cd OneR && python unsw.py`                            |

*For CIC-IDS2017: replace `unsw` with `dcic2017` or `cicids2017`.*

---

## Ablation Studies
**UNSW-NB15 Variants:**
```bash
cd Ablation_Study

python full_has_ids_unsw.py                      # Full model
python has-ids_unsw_wo_prob.py                   # Without BGMM
python has-ids_unsw_wo_contextualScores.py       # Without contextual scoring
python has-ids_unsw_wo_instanceScore.py          # Without ANN
python has-ids_unsw_probOnly.py                  # BGMM only
python has-ids_unsw_contextOnly.py               # Contextual only
python has-ids_unsw_AnnOnly_woBgmm.py            # ANN only
```

*For CIC-IDS2017: replace `unsw` with `cicids2017` or `ids2017`.*

---

## Hyperparameter Configuration

### UNSW-NB15

| Parameter                | Value         |
|--------------------------|---------------|
| Encoder architecture     | 196-128-64-32 |
| Feature dimension        | 32            |
| SCL temperature          | 0.38          |
| Epochs                   | 18            |
| Batch size               | 128           |
| Learning rate            | 0.0018        |
| Optimizer                | Adam          |
| BGMM components          | 5             |
| BGMM prior               | 0.01          |
| Trimmed-mean fraction    | 0.10          |
| Coverage                 | 0.97          |
| FAISS neighbours         | 7             |
| Guard quantile           | 0.995         |
| Target FPR               | 0.10          |

### CIC-IDS2017

| Parameter                | Value          |
|--------------------------|----------------|
| Encoder architecture     | 82-256-128-128 |
| Feature dimension        | 128            |
| SCL temperature          | 0.05           |
| Epochs                   | 50             |
| Batch size               | 256            |
| Learning rate            | 0.0010         |
| Optimizer                | Adam           |
| BGMM components          | 3              |
| BGMM prior               | 0.004          |
| Trimmed-mean fraction    | 0.10           |
| Coverage                 | 0.97           |
| FAISS neighbours         | 7              |
| Guard quantile           | 0.995          |
| Target FPR               | 0.10           |

---

## Custom Dataset Evaluation

**Prepare CSV data** with numerical features and binary labels (0=normal, 1=attack).

**Split data:**
```bash
cd Data_split && python split.py
```

**Update paths** in `has_ids_unsw.py` or `has_ids_ids2017.py`:
```python
train_path = "../Datasets/Custom/custom_train.csv"
test_path = "../Datasets/Custom/custom_test.csv"
```

**Adjust hyperparameters** as needed (see configuration table above).

---

**Version**: 1.0.0 | **Updated**: February 4, 2026
