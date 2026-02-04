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
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, faiss, sklearn, pandas; print('Ready')"

# Download and prepare datasets (see Dataset Preparation section below)

# Run first experiment
cd HAS-IDS
python has_ids_unsw.py
```

---

## Dataset Preparation

**Datasets are NOT included in this repository.** You must download them separately:

### 1. Download Datasets

**UNSW-NB15:**
- Official: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Download pre-processed CSV files or raw PCAP files

**CIC-IDS2017 (Corrected Version):**
- DistriNet (Recommended): https://intrusion-detection.distrinet-research.be/WTMC2021/
- Kaggle Mirror: https://www.kaggle.com/datasets/dhoogla/distrinetcicids2017/data

### 2. Create Directory Structure

Create the following structure in the project root:
```
Datasets/
├── UNSW/
│   ├── BUNSWTrain.csv          # Binary training set
│   ├── BUNSWTest.csv           # Binary test set
│   ├── MUNSWTrain.csv          # Multiclass training set
│   └── MUNSWTest.csv           # Multiclass test set
└── DCIC2017/
    ├── DBcic2017_train.csv     # Binary training set
    ├── DBcic2017_test.csv      # Binary test set
    ├── DMcic2017_train.csv     # Multiclass training set
    └── DMcic2017_test.csv      # Multiclass test set
```

### 3. Preprocess Raw Data (Optional)

If you downloaded raw data, use the preprocessing scripts:
```bash
cd Dataset_split

# For CIC-IDS2017
python Dcicids2017.py

# For general train/test split
python split.py
python DBsplit.py  # Binary split
python DMsplit.py  # Multiclass split
```

---

## Main Experiments

**UNSW-NB15:**
```bash
cd HAS-IDS
python has_ids_unsw.py
```

**CIC-IDS2017:**
```bash
cd HAS-IDS
python has_ids_dcic.py
```

**Output Files:**
- Model checkpoint: `*.joblib`
- Predictions: `*_predictions.csv`
- Metrics: `*_metrics.json`

---

## Baseline Methods

| Method              | Command (UNSW-NB15)                                          | Command (CIC-IDS2017)                                     |
|---------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| **CL-BGMM**         | `cd baselines/CL-BGMM && python cl_bgmm_unsw.py`             | `cd baselines/CL-BGMM && python cl_bgmm_dcic.py`          |
| **AOC-IDS**         | `cd baselines/aocids && python run_aoc_ids_unsw.py`          | *(UNSW only)*                                             |
| **CIDS**            | `cd baselines/CIDS && python cids_unsw.py`                   | *(UNSW only)*                                             |
| **Isolation Forest**| `cd baselines/Isolation_forest && python isolation_forest_unsw.py` | `cd baselines/Isolation_forest && python isolation_forest_dcic.py` |
| **Autoencoder**     | `cd baselines/autoencoder && python run_unsw_ae.py`          | `cd baselines/autoencoder && python run_dcic_ae.py`       |

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

**CIC-IDS2017 Variants:**
```bash
cd Ablation_Study

python full-has_ids_cicids2017.py                # Full model
python has-ids_cicids2017_wo_prob.py             # Without BGMM
python has-ids_ids2017_wo_contextualScores.py    # Without contextual scoring
python has-ids_ids2017_wo_instanceScore.py       # Without ANN
python has-ids_ids2017_probOnly.py               # BGMM only
python has-ids_cicids2017_contextOnly.py         # Contextual only
python has-ids_ids2017_AnnOnly_woBgmm.py         # ANN only
```

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
