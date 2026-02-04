# HAS-IDS: Hybrid Anomaly Scoring for Low-Frequency Network Intrusion Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-red.svg)](https://pytorch.org/)

**Authors:** Salah Abdullah Khalil Abdulrahman, Vera Suryani  
**Affiliation:** School of Computing, Telkom University, Bandung 40257, Indonesia  
**Contact:** salahabdullahkhalil@student.telkomuniversity.ac.id

---

## 📄 Abstract

Network intrusion detection systems (NIDS) face severe class imbalance: low-frequency attacks are overshadowed by benign traffic and a few high-volume families, so models can score well on aggregate metrics while missing rare but critical threats. **HAS-IDS** is proposed as a **Hybrid Anomaly Scoring Intrusion Detection System** that couples supervised contrastive representation learning with calibrated hybrid anomaly scoring under a fixed false-positive budget.

A supervised contrastive encoder learns class-discriminative flow embeddings. A meta-classifier fuses three evidence streams:
1. **Probabilistic normality** from a Bayesian Gaussian mixture model over benign scores
2. **Instance-based attack similarity** via approximate nearest neighbor search in the embedding space
3. **Mixture component context statistics**

Operating thresholds are selected on a held-out calibration split to keep the calibration false-positive rate within a user-specified budget α and are then fixed for testing.

### Key Results
- **UNSW-NB15**: 93.67% F₁-score at 9.93% FPR (vs. CL-BGMM: 92.35% F₁ at 11.61% FPR)
- **Distrinet-CIC-IDS2017**: 99.80% F₁-score at 0.09% FPR with significant recall improvements on ultra-rare attacks:
  - Infiltration: 100.00% (vs. 25.00%)
  - Portscan: 99.40% (vs. 78.38%)
  - SQL Injection: 66.67% (vs. 33.33%)
- **PT XYZ-2025** (0.38% attack prevalence): Detects attacks at ≈0.2% FPR

---

## 🎯 Key Features

- ✅ **Hybrid Anomaly Scoring**: Combines probabilistic, instance-based, and contextual evidence
- ✅ **Symmetric Supervised Contrastive Learning**: Preserves inter-attack topology in embedding space
- ✅ **Calibrated False-Positive Control**: Fixed FPR budget for practical SOC deployment
- ✅ **Excellent Rare-Attack Detection**: Dramatically improved recall on ultra-rare classes (<0.01% prevalence)
- ✅ **Comprehensive Evaluation**: Tested on UNSW-NB15, CIC-IDS2017, and enterprise data
- ✅ **Modular Architecture**: Easy to extend and customize individual components
- ✅ **GPU/CPU Flexibility**: Automatic device detection with FAISS support

---

## 📁 Project Structure

```
HAS-IDS/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # License information
├── CITATION.bib                 # Citation information
│
├── HAS_IDS/                     # ⭐ Main Implementation
│   ├── has_ids_unsw.py          # HAS-IDS for UNSW-NB15
│   ├── has_ids_ids2017.py       # HAS-IDS for CIC-IDS2017
│   └── eval_classes.py          # Per-class recall evaluation
│
├── Ablation_Study/              # Ablation Study Variants (14 files)
│   ├── full_has_ids_unsw.py     # Full model (UNSW)
│   ├── full-has_ids_cicids2017.py  # Full model (CIC-IDS2017)
│   ├── has-ids_*_wo_prob.py     # Without probabilistic stream
│   ├── has-ids_*_wo_contextualScores.py  # Without context
│   ├── has-ids_*_wo_instanceScore.py     # Without ANN
│   ├── has-ids_*_probOnly.py    # Probabilistic stream only
│   ├── has-ids_*_contextOnly.py # Context only
│   └── has-ids_*_AnnOnly_woBgmm.py  # Instance-based only
│
├── Datasets/                    # Dataset Storage
│   ├── DCIC2017/                # CIC-IDS2017 Splits
│   │   ├── DBcic2017_train.csv  # Binary train
│   │   ├── DBcic2017_test.csv   # Binary test
│   │   ├── DMcic2017_train.csv  # Multiclass train
│   │   └── DMcic2017_test.csv   # Multiclass test
│   │
│   └── UNSW/                    # UNSW-NB15 Splits
│       ├── BUNSWTrain.csv       # Binary train
│       ├── BUNSWTest.csv        # Binary test
│       ├── MUNSWTrain.csv       # Multiclass train
│       └── MUNSWTest.csv        # Multiclass test
│
├── Data_split/                  # Data Preprocessing Scripts
│   ├── split.py                 # General train/test split
│   ├── DBsplit.py               # Binary split
│   ├── DMsplit.py               # Multiclass split
│   └── Dcicids2017.py           # CIC-IDS2017 specific preprocessing
│
├── baselines/                   # Baseline Method Wrappers
│   └── aocids/                  # AOC-IDS wrapper scripts (upstream: github.com/xinchen930/AOC-IDS)
│       ├── run_aoc_ids_unsw.py  # Main script with 5-seed evaluation
│       ├── utils_aoc.py         # Utility functions
│       ├── eval_classes.py      # Per-class recall evaluation
│       ├── eval_classes_averager.py  # Result averaging
│       └── README.md            # Usage instructions
│
├── CL-BGMM/                     # Baseline: CL-BGMM Implementation
│   ├── bgmm_unsw_bv2_fixedv2.py
│   ├── bgmm_cic_v2_fixedv2.py
│   └── eval_unsw_v2.py
│
├── CIDS/                        # Baseline: CIDS Implementation
│   └── cids_unsw.py
│
├── Isolation_forest/            # Baseline: Isolation Forest
│   ├── isolation_forest_unsw.py
│   └── isolation_forest_dcic2017.py
│
├── OneR/                        # Baseline: OneR
│   ├── unsw.py
│   └── dcic2017.py
│
├── autoencoder/                 # Baseline: Autoencoder
│   ├── run_unsw_ae.py
│   └── run_dcic2017_ae.py
│
└── Visual_t-SNE_histo/          # Visualization Scripts
    ├── has-ids_unsw_vis.py      # t-SNE & histogram for UNSW
    └── has-ids_dcic_vis.py      # t-SNE & histogram for CIC-IDS2017
```

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.7 or higher (optional, for GPU acceleration)
- **Operating System**: Linux, Windows, or macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/HAS-IDS.git
cd HAS-IDS
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n hasids python=3.8
conda activate hasids

# OR using venv
python -m venv hasids_env
source hasids_env/bin/activate  # On Windows: hasids_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch==1.13.1` (or compatible version)
- `numpy==1.23.5`
- `pandas==1.5.3`
- `scikit-learn==1.2.1`
- `scipy==1.10.0`
- `joblib==1.2.0`
- `faiss-cpu==1.7.4` (or `faiss-gpu` for GPU acceleration)
- `matplotlib==3.6.0`
- `seaborn==0.12.0`

### Step 4: Verify Installation

```bash
python -c "import torch; import faiss; print('Installation successful!')"
```

---

## 📊 Datasets

### Supported Datasets

1. **UNSW-NB15**
   - **Source**: [UNSW Sydney](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - **Description**: 196-dimensional flow features, binary and multiclass labels
   - **Size**: ~2.5M flows with 10 attack categories

2. **Distrinet-CIC-IDS2017 (Corrected)**
   - **Source**: [DistriNet Research](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html)
   - **Alternative**: [Kaggle Mirror](https://www.kaggle.com/datasets/dhoogla/distrinetcicids2017/data)
   - **Description**: 82-dimensional flow features, corrected labels
   - **Size**: ~2.8M flows with 15 attack categories

3. **PT XYZ-2025 (Enterprise)**
   - **Description**: Proprietary enterprise dataset with extreme imbalance (0.38% attack prevalence)
   - **Note**: Raw data not publicly available due to privacy restrictions
   - **Access**: Contact authors for de-identified version

### Dataset Preparation

#### Option 1: Download Pre-processed Datasets

Place the CSV files in the appropriate directories:

```
Datasets/
├── DCIC2017/
│   ├── DBcic2017_train.csv
│   ├── DBcic2017_test.csv
│   ├── DMcic2017_train.csv
│   └── DMcic2017_test.csv
└── UNSW/
    ├── BUNSWTrain.csv
    ├── BUNSWTest.csv
    ├── MUNSWTrain.csv
    └── MUNSWTest.csv
```

#### Option 2: Generate Splits from Raw Data

If you have the raw datasets, use the preprocessing scripts:

```bash
# For CIC-IDS2017
cd Data_split
python Dcicids2017.py

# For general train/test split
python split.py
```

---

## 🚀 Usage

### Quick Start: Run HAS-IDS

#### 1. Run on UNSW-NB15

```bash
cd HAS_IDS
python has_ids_unsw.py
```

**Expected output:**
- Training logs with epoch progress
- Test set performance metrics (Accuracy, Precision, Recall, F1, FPR, ROC-AUC, PR-AUC)
- Saved artifacts: `hasids_unsw_nb15_artifacts.joblib`
- Predictions: `unsw_nb15_hasids_test_predictions.csv`
- Metrics: `unsw_nb15_hasids_test_metrics.json`

#### 2. Run on CIC-IDS2017

```bash
cd HAS_IDS
python has_ids_ids2017.py
```

**Expected output:**
- Training logs with epoch progress
- Test set performance metrics
- Saved artifacts: `cicids2017_hasids_faiss_ann_artifacts.joblib`
- Predictions: `cicids2017_hasids_test_predictions.csv`
- Metrics: `cicids2017_hasids_test_metrics.json`

### Evaluate Per-Class Performance

```bash
cd HAS_IDS
python eval_classes.py --binary path/to/predictions.csv --multi path/to/multiclass_labels.csv
```

### Configuration Options

Edit the configuration variables at the top of each script:

```python
# Device selection
PREFERRED_TORCH_DEVICE = "cuda"  # "auto" | "cuda" | "cpu"
PREFERRED_FAISS_DEVICE = "cpu"   # "auto" | "gpu" | "cpu"

# Hyperparameters (example)
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
TEMPERATURE = 0.4
```

---

## 🧪 Experiments

### Run Baseline Methods

#### CL-BGMM Baseline

```bash
cd CL-BGMM
python bgmm_unsw_bv2_fixedv2.py  # For UNSW-NB15
python bgmm_cic_v2_fixedv2.py     # For CIC-IDS2017
```

#### AOC-IDS Baseline

**Note:** AOC-IDS wrapper scripts are in `baselines/aocids/`. The upstream AOC-IDS code is from https://github.com/xinchen930/AOC-IDS (Zhang et al., 2024). Our scripts add 5-seed averaging, extended metrics (ROC-AUC, PR-AUC, FPR, per-class recall), and computation cost tracking.

```bash
cd baselines/aocids
python run_aoc_ids_unsw.py
```

See [baselines/aocids/README.md](baselines/aocids/README.md) for details.

#### CIDS Baseline

```bash
cd CIDS
python cids_unsw.py --train ../Datasets/UNSW/BUNSWTrain.csv --test ../Datasets/UNSW/BUNSWTest.csv
```

#### Isolation Forest

```bash
cd Isolation_forest
python isolation_forest_unsw.py
python isolation_forest_dcic2017.py
```

#### Autoencoder

```bash
cd autoencoder
python run_unsw_ae.py
python run_dcic2017_ae.py
```

### Run Ablation Studies

```bash
cd Ablation_Study

# Full model
python full_has_ids_unsw.py

# Without probabilistic stream
python has-ids_unsw_wo_prob.py

# Without contextual scores
python has-ids_unsw_wo_contextualScores.py

# Without instance-based scoring
python has-ids_unsw_wo_instanceScore.py

# Probabilistic stream only
python has-ids_unsw_probOnly.py

# Context only
python has-ids_unsw_contextOnly.py

# ANN only (without BGMM)
python has-ids_unsw_AnnOnly_woBgmm.py
```

### Visualization

```bash
cd Visual_t-SNE_histo

# Generate t-SNE and histogram plots for UNSW-NB15
python has-ids_unsw_vis.py

# Generate visualizations for CIC-IDS2017
python has-ids_dcic_vis.py
```

---

## 📈 Expected Results

### UNSW-NB15

| Metric | HAS-IDS | CL-BGMM | CIDS | Improvement |
|--------|---------|---------|------|-------------|
| F₁-Score | 93.67% | 92.35% | 82.38% | **+1.32 pp** |
| Precision | 91.59% | 87.45% | 71.86% | **+4.14 pp** |
| Recall | 95.82% | 97.96% | 97.61% | -2.14 pp |
| FPR | 9.93% | 11.61% | 30.08% | **-1.68 pp** |
| ROC-AUC | 97.87% | 97.94% | 96.48% | -0.07 pp |

### Distrinet-CIC-IDS2017

| Metric | HAS-IDS | CL-BGMM | Improvement |
|--------|---------|---------|-------------|
| F₁-Score | 99.80% | 99.60% | **+0.20 pp** |
| Precision | 99.71% | 99.48% | **+0.23 pp** |
| Recall | 99.88% | 99.72% | **+0.16 pp** |
| FPR | 0.09% | 0.12% | **-0.03 pp** |
| ROC-AUC | 99.95% | 99.98% | -0.03 pp |

### Rare-Attack Recall (CIC-IDS2017)

| Attack Class | HAS-IDS | CL-BGMM | Improvement |
|--------------|---------|---------|-------------|
| Infiltration | **100.00%** | 25.00% | **+75.00 pp** |
| Portscan | **99.40%** | 78.38% | **+21.02 pp** |
| SQL Injection | **66.67%** | 33.33% | **+33.34 pp** |

---

## ⚙️ Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **GPU**: Not required (CPU mode available)

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16 GB
- **Storage**: 20 GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3090 or equivalent)
- **CUDA**: 11.7+

### Runtime Estimates

#### UNSW-NB15 (82,332 test flows)
- **Training Time**: ~130 seconds (with GPU)
- **Test Time**: ~1.3 seconds
- **Per-flow Inference**: ~0.016 ms

#### CIC-IDS2017 (287,328 test flows)
- **Training Time**: ~1,323 seconds (with GPU)
- **Test Time**: ~2.9 seconds
- **Per-flow Inference**: ~0.010 ms

*Tested on: NVIDIA RTX 3090, Intel i9-10900K, 32GB RAM*

---

## 📊 Output Files

### Artifacts Saved by HAS-IDS

1. **Model Artifacts** (`.joblib`)
   - Encoder state dictionary
   - Scaler
   - BGMM model
   - Meta-classifier
   - Optimal thresholds
   - Attack bank
   - Feature column names

2. **Predictions** (`.csv`)
   - Columns: `binary_label`, `binary_pred`, `attack_prob`
   - One row per test sample

3. **Metrics** (`.json`)
   - Accuracy, Precision, Recall, F₁-Score, FPR
   - ROC-AUC, PR-AUC
   - Training and test time
   - Confusion matrix

4. **Visualizations** (`.png`)
   - t-SNE embeddings
   - Score histograms
   - Confusion matrices

---

## 🔬 Ablation Study Results

| Variant | F₁ (UNSW) | FPR (UNSW) | F₁ (CIC) | FPR (CIC) |
|---------|-----------|------------|----------|-----------|
| **Full HAS-IDS** | 93.67% | 9.93% | 99.80% | 0.09% |
| Probabilistic Only | **93.72%** | **9.88%** | 99.59% | 0.13% |
| w/o Context | 93.51% | 10.08% | 99.53% | **0.42%** ↑ |
| w/o Instance | 93.64% | 9.91% | **99.80%** | **0.09%** |
| w/o Probabilistic | 92.94% | 10.53% | 99.18% | 0.25% |
| Context Only | 92.85% | 10.69% | 99.12% | 0.31% |
| Instance Only | 75.39% | 25.78% | 97.84% | 0.68% |

**Key Findings:**
- ✅ Probabilistic stream (BGMM) is the primary performance driver
- ✅ Contextual statistics stabilize FPR in long-tailed regimes
- ⚠️ Instance-based similarity provides marginal gains at higher cost
- ❌ Instance-only mode fails catastrophically on rare classes

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size in the script
BATCH_SIZE = 256  # Default: 512
```

#### 2. FAISS Not Found

**Error:** `ModuleNotFoundError: No module named 'faiss'`

**Solution:**
```bash
# Install FAISS CPU version
pip install faiss-cpu

# OR install GPU version
conda install -c pytorch faiss-gpu
```

#### 3. File Not Found (Hardcoded Paths)

**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:** Ensure datasets are in the correct location:
```bash
ls Datasets/UNSW/BUNSWTrain.csv
ls Datasets/DCIC2017/DBcic2017_train.csv
```

#### 4. Slow Inference on CPU

**Issue:** Test time is very slow

**Solution:**
```python
# Enable GPU if available
PREFERRED_TORCH_DEVICE = "cuda"  # Instead of "cpu"
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{abdulrahman2026hasids_code,
  author={Abdulrahman, Salah Abdullah Khalil and Suryani, Vera},
  title={HAS-IDS: Hybrid Anomaly Scoring for Low-Frequency Network Intrusion Detection - Implementation},
  year={2026},
  publisher={GitHub},
  url={https://github.com/yourusername/HAS-IDS},
  version={1.0.0}
}
```

**Note:** Paper citation will be added here after publication. The accompanying research paper is currently under preparation for submission to PeerJ Computer Science.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Authors

- **Salah Abdullah Khalil Abdulrahman** - *Lead Author* - [Email](mailto:salahabdullahkhalil@student.telkomuniversity.ac.id)
- **Vera Suryani** - *Supervisor* - School of Computing, Telkom University

---

## 🙏 Acknowledgments

- Telkom University, Indonesia, for providing research facilities and institutional support
- Forensic and Network Security (FORESTY) Research Laboratory for technical assistance
- The authors of CL-BGMM, AOC-IDS, and CIDS for providing baseline implementations
- UNSW Sydney and DistriNet for maintaining public intrusion detection datasets

---

## 📞 Contact

For questions, bug reports, or collaboration inquiries:

- **Email**: salahabdullahkhalil@student.telkomuniversity.ac.id
- **Institution**: School of Computing, Telkom University, Bandung 40257, Indonesia
- **Research Group**: Forensic and Network Security (FORESTY) Laboratory

---

## 🔗 Links

- **Paper**: [PeerJ Submission](https://peerj.com) *(Under Review)*
- **Code Repository**: https://github.com/yourusername/HAS-IDS
- **UNSW-NB15 Dataset**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CIC-IDS2017 Dataset**: https://intrusion-detection.distrinet-research.be/WTMC2021/
- **Code Archive**: [DOI will be added upon publication]

---

## 📝 Version History

- **v1.0.0** (2026-02-04): Initial release for PeerJ submission
  - Core HAS-IDS implementation
  - 6 baseline methods
  - 14 ablation variants
  - Comprehensive evaluation on 3 datasets

---

## ⚖️ Disclaimer

This software is provided for research purposes only. The authors are not responsible for any misuse or damage caused by this software. Always comply with local laws and regulations when deploying intrusion detection systems.

---

**Last Updated**: February 4, 2026  
**Status**: Ready for PeerJ Submission ✅
