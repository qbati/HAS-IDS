# HAS-IDS Usage Guide

Concise instructions for running HAS-IDS experiments and reproducing the reported results.

**Version:** 1.0.3  
**Updated:** May 2026

---

## 1. Environment Setup

Install the required Python packages from the project root:

```bash
pip install -r requirements.txt
```

Quick check:

```bash
python -c "import torch, sklearn, pandas, numpy; print('Ready')"
```

FAISS is optional but recommended. If FAISS is not available, the code falls back to NumPy-based similarity search where implemented.

---

## 2. Dataset Preparation

Datasets are not included in this repository. Download them from the original public sources and place the prepared CSV files under the `Datasets/` directory.

### UNSW-NB15

Download UNSW-NB15 from the official source:

```text
https://research.unsw.edu.au/projects/unsw-nb15-dataset
```

Expected files:

```text
Datasets/UNSW/
├── BUNSWTrain.csv
├── BUNSWTest.csv
├── MUNSWTrain.csv
└── MUNSWTest.csv
```

`BUNSW*.csv` files are binary-label splits.  
`MUNSW*.csv` files contain the corresponding multiclass labels used for per-class recall reporting.

### Corrected Distrinet-CIC-IDS2017

Download the corrected Distrinet-CIC-IDS2017 dataset from:

```text
https://intrusion-detection.distrinet-research.be/WTMC2021/
```

or the Kaggle mirror:

```text
https://www.kaggle.com/datasets/dhoogla/distrinetcicids2017/data
```

Expected files:

```text
Datasets/DCIC2017/
├── DBcic2017_train.csv
├── DBcic2017_test.csv
├── DMcic2017_train.csv
└── DMcic2017_test.csv
```

`DBcic2017_*.csv` files are binary-label splits.  
`DMcic2017_*.csv` files contain the corresponding multiclass labels used for per-class recall reporting.

### Split seed

The manuscript uses fixed stratified train/test partitions. The split seed is:

```text
random_state = 42
```

---

## 3. Running the Main Experiments

Run the scripts from inside the `HAS-IDS/` directory.

```bash
cd HAS-IDS
```

### UNSW-NB15

```bash
python has_ids_unsw.py
```

Main outputs:

```text
unsw_nb15_hasids_test_predictions.csv
unsw_nb15_hasids_test_metrics.json
hasids_unsw_nb15_artifacts.joblib
```

### Corrected Distrinet-CIC-IDS2017

```bash
python has_ids_dcic.py
```

Main outputs:

```text
cicids2017_hasids_test_predictions.csv
cicids2017_hasids_test_metrics.json
cicids2017_hasids_faiss_ann_artifacts.joblib
```

---

## 4. Per-Class Recall Evaluation

Per-class recall is computed by aligning binary predictions with the multiclass test-label files.

Use:

```bash
python eval_classes.py
```

Before running, edit the paths inside `eval_classes.py` so they point to:

```text
prediction CSV
multiclass test CSV
```

For Normal traffic, the reported value is specificity.  
For attack classes, the reported value is recall.

---

## 5. Ten-Run Reproducibility Evaluation

The revised manuscript reports mean ± standard deviation for HAS-IDS over 10 independent runs.

Run these scripts from inside the `HAS-IDS/` directory:

```bash
cd HAS-IDS
```

### UNSW-NB15, Tables 6 and 8

```bash
python run_10seeds_unsw.py
```

Seed list:

```text
42, 0, 1, 2, 3, 4, 5, 6, 7, 8
```

Outputs:

```text
unsw_10seeds_raw_results.csv
unsw_10seeds_summary.txt
```

### Corrected Distrinet-CIC-IDS2017, Tables 10 and 12

```bash
python run_10seeds_dcic.py
```

Seed list:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

Outputs:

```text
dcic_10seeds_raw_results.csv
dcic_10seeds_summary.txt
```

The summary files include both per-seed results and LaTeX-ready mean ± std values for the manuscript tables.

Before running either script, check the `CONFIGURE ONLY THESE` block at the top of the file and update dataset paths if your local directory structure differs.

---

## 6. Ablation Study

Ablation scripts are stored in:

```text
Ablation_Study/
```

Each script corresponds to one evidence-stream variant, such as:

```text
Full HAS-IDS
without BGMM/probabilistic stream
without contextual stream
without instance/ANN stream
probability-only
context-only
ANN-only
```

Important note for corrected Distrinet-CIC-IDS2017:

```text
The main CIC evaluation uses learning rate η = 0.0010.
The CIC ablation study uses diagnostic learning rate η = 0.0020.
```

This diagnostic setting is used to keep stream-contribution differences visible in the ablation tables. The main benchmark results remain those reported by `has_ids_dcic.py` and `run_10seeds_dcic.py`.

---

## 7. Baseline Methods

Baseline implementations are stored in:

```text
baselines/
```

The repository includes baseline folders for:

```text
CL-BGMM
AOC-IDS
CIDS
Autoencoder
Isolation Forest
```

Run each baseline from its corresponding folder. Some baselines are dataset-specific; check the script names and folder-level comments before execution.

---

## 8. Key Hyperparameters

### UNSW-NB15

| Parameter | Value |
|---|---:|
| Encoder | 196-128-64-32 |
| Feature dimension | 32 |
| SCL temperature | 0.38 |
| Epochs | 18 |
| Batch size | 128 |
| Learning rate | 0.0018 |
| Optimizer | Adam |
| BGMM setting | K = 5, reg = 1e-2 |
| FAISS neighbours | 7 |
| Guard quantile | 0.995 |
| Target FPR | 0.10 |

### Corrected Distrinet-CIC-IDS2017

| Parameter | Value |
|---|---:|
| Encoder | 82-256-128-128 |
| Feature dimension | 128 |
| SCL temperature | 0.05 |
| Epochs | 50 |
| Batch size | 256 |
| Learning rate | 0.0010 |
| Optimizer | Adam |
| BGMM setting | K = 3, reg = 4e-3 |
| FAISS neighbours | 7 |
| Guard quantile | 0.995 |
| Target FPR | 0.10 |

---

## 9. Notes on Threshold Search

For corrected Distrinet-CIC-IDS2017, the v1.0.3 code evaluates threshold candidates using the observed calibration scores directly, rather than a rounded score grid.

This avoids rounding-induced degenerate all-Normal predictions in saturated multi-seed runs while keeping the FPR constraint and tie-breaking logic unchanged.

---

## 10. Hardware Used for Reported Runs

The reported experiments were run on:

```text
CPU: Intel Core i7-12700H
RAM: 16 GB
GPU: NVIDIA RTX 3070 Ti, 8 GB VRAM
```

Approximate runtime on this hardware:

| Script | Dataset | Approx. runtime |
|---|---|---:|
| `has_ids_unsw.py` | UNSW-NB15 | ~8 min |
| `has_ids_dcic.py` | Corrected Distrinet-CIC-IDS2017 | ~25 min |
| `run_10seeds_unsw.py` | UNSW-NB15 × 10 | ~80 min |
| `run_10seeds_dcic.py` | Corrected Distrinet-CIC-IDS2017 × 10 | ~250 min |

Runtime may vary depending on hardware, FAISS availability, and GPU configuration.

---

## 11. Recommended Execution Order

For a full reproduction workflow:

```bash
# 1. Prepare datasets under Datasets/

# 2. Run main single-seed experiments
cd HAS-IDS
python has_ids_unsw.py
python has_ids_dcic.py

# 3. Run 10-seed reproducibility experiments
python run_10seeds_unsw.py
python run_10seeds_dcic.py

# 4. Run per-class evaluation if needed
python eval_classes.py
```

---

## 12. Citation

Use `CITATION.bib` for citation metadata. The Zenodo DOI should be updated after the v1.0.3 archive is published.
