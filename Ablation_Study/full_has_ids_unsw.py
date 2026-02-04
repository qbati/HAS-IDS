import os
# ===== Quick switches (edit these) ===========================================
PREFERRED_TORCH_DEVICE = "cuda"   # "auto" | "cuda" | "cpu"
PREFERRED_FAISS_DEVICE = "cpu"    # "auto" | "gpu"  | "cpu"
# ============================================================================

# (Optional) cap CPU threading to keep sklearn / faiss-CPU predictable
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import random
import warnings
import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Try to import FAISS (CPU or GPU). We'll handle fallback if not present.
_FAISS_AVAILABLE = False
try:
    import faiss  # pip install faiss-cpu  (or faiss-gpu)
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


# ----------------------------------------------------------------------------
# 1. UTILITY FUNCTIONS
# ----------------------------------------------------------------------------

def setup_seed(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSeed {seed} set for reproducibility.")


def score_detail(y_true, y_pred, y_score=None, title="Evaluation Results"):
    """Calculates and prints a comprehensive set of performance metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # ensure 2x2
    if cm.size != 4:
        cm = np.zeros((2, 2), dtype=int)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"\n--- {title} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            roc = roc_auc_score(y_true, y_score)
            pr = average_precision_score(y_true, y_score)
            print(f"ROC-AUC:   {roc:.4f}")
            print(f"PR-AUC:    {pr:.4f}")
        except Exception:
            pass
    return acc, prec, rec, f1, fpr


def _cm_2x2(y_true, y_pred):
    """Ensures confusion matrix is always 2x2 for stable metric calculation."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        pad = np.zeros((2, 2), dtype=int)
        pad[: cm.shape[0], : cm.shape[1]] = cm
        cm = pad
    return cm.ravel()


# ----------------------------------------------------------------------------
# 2. DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------------

def load_and_preprocess_unsw_nb15(train_path, test_path):
    """Loads and preprocesses UNSW-NB15 with no-leakage one-hot encoding."""
    df_train_orig = pd.read_csv(train_path)
    df_test_orig = pd.read_csv(test_path)
    df_train, df_test = df_train_orig.copy(), df_test_orig.copy()
    for df in (df_train, df_test):
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    # Expect 'label' column exists and is binary (0 normal, 1 attack)
    y_train = df_train["label"].values.astype(int)
    y_test = df_test["label"].values.astype(int)

    feature_cols = [c for c in df_train.columns if c not in ["label", "attack_cat"]]
    cat_cols = [c for c in feature_cols if df_train[c].dtype == "object"]
    num_cols = [c for c in feature_cols if df_train[c].dtype != "object"]

    X_train = pd.get_dummies(df_train[feature_cols], columns=cat_cols)
    X_test = pd.get_dummies(df_test[feature_cols], columns=cat_cols)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = MinMaxScaler()
    if len(num_cols) > 0:
        # only transform numeric columns; get_dummies may have created new columns for categorical
        numeric_intersect = [c for c in num_cols if c in X_train.columns]
        if len(numeric_intersect) > 0:
            X_train[numeric_intersect] = scaler.fit_transform(X_train[numeric_intersect])
            X_test[numeric_intersect] = scaler.transform(X_test[numeric_intersect])

    print(f"Preprocessing complete. Feature shape: {X_train.shape}")
    return (
        X_train.values.astype(np.float32),
        y_train,
        X_test.values.astype(np.float32),
        y_test,
        df_test_orig,
        scaler,
        feature_cols,
        num_cols,
        cat_cols,
    )


# ----------------------------------------------------------------------------
# 3. DEEP FEATURE EXTRACTOR AND LOSS FUNCTION
# ----------------------------------------------------------------------------

class AttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=32, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.head = nn.Linear(64, feature_dim)

    def forward(self, x):
        attn = F.softmax(self.attention(x), dim=1)
        h = self.backbone(x * attn)
        return self.head(h)


class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.4):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        z = F.normalize(features, p=2, dim=1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = (z @ z.T) / self.temperature
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob_pos.mean()


# ----------------------------------------------------------------------------
# 4. HAS-IDS DECISION FRAMEWORK HELPERS
# ----------------------------------------------------------------------------

def fit_bgmm_on_scores(z_scores_np, n_components=3, seed=42, reg_covar=4e-3):
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type="dirichlet_process",
        max_iter=500,
        n_init=1,
        random_state=seed,
        reg_covar=reg_covar,
    )
    bgmm.fit(z_scores_np.reshape(-1, 1))
    return bgmm


def map_components_to_labels_coverage(bgmm, z_scores_np, y_np, cover_gamma=0.97):
    resp = bgmm.predict_proba(z_scores_np.reshape(-1, 1))
    lab = np.array(["attack"] * bgmm.n_components, dtype=object)
    normal_idx = (y_np == 0)
    if normal_idx.sum() == 0:
        lab[np.argmax(bgmm.means_.flatten())] = "normal"
        return lab
    mass_per_k = resp[normal_idx].sum(axis=0)
    total_mass = mass_per_k.sum()
    if total_mass <= 0:
        lab[np.argmax(bgmm.means_.flatten())] = "normal"
        return lab
    order = np.argsort(bgmm.means_.flatten())[::-1]  # descending by mean
    cum = 0.0
    for k in order:
        lab[k] = "normal"
        cum += mass_per_k[k]
        if cum / total_mass >= cover_gamma:
            break
    return lab


def posterior_normal(bgmm, z_scores_np, comp_label):
    resp = bgmm.predict_proba(z_scores_np.reshape(-1, 1))
    normal_mask = (comp_label == "normal")
    if normal_mask.sum() == 0:
        return np.zeros(len(z_scores_np), dtype=np.float32), resp
    return resp[:, normal_mask].sum(axis=1), resp


# ------------------------------- FAISS ANN -----------------------------------

def _normalize_rows_l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows of x in-place-safe manner."""
    x = x.astype(np.float32, copy=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_faiss_index(
    bank_vecs: np.ndarray,
    method: str = "flatip",
    nlist: int = 256,
    pq_m: int = 16,
    hnsw_m: int = 32,
    use_gpu: bool = False,
):
    """
    Build a FAISS index over bank vectors.
    - method: "flatip" (fast, exact), "ivfpq" (coarse quantized), "hnsw" (graph-based)
    - All methods assume vectors are L2-normalized and use Inner Product (cosine sim).
    """
    if bank_vecs is None or bank_vecs.shape[0] == 0:
        return None

    if not _FAISS_AVAILABLE:
        return None  # Fallback handled by caller

    vecs = _normalize_rows_l2(bank_vecs)
    d = vecs.shape[1]

    if method == "flatip":
        index = faiss.IndexFlatIP(d)

    elif method == "ivfpq":
        quantizer = faiss.IndexFlatIP(d)
        # Ensure nlist is reasonable
        nlist_eff = max(1, min(nlist, vecs.shape[0]))
        index = faiss.IndexIVFPQ(quantizer, d, nlist_eff, pq_m, 8)  # 8 bits/code
        # train requires some samples and float32
        train_n = min(50000, len(vecs))
        if train_n > 0:
            train_sample = vecs[np.random.choice(len(vecs), train_n, replace=False)].astype(np.float32)
            try:
                index.train(train_sample)
            except Exception as e:
                # If training fails, fall back to a flat index
                print(f"Warning: IVFPQ train failed: {e}. Falling back to IndexFlatIP.")
                index = faiss.IndexFlatIP(d)

    elif method == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)

    else:
        raise ValueError(f"Unknown FAISS method: {method}")

    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"Warning: failed to move FAISS index to GPU: {e}. Using CPU index.")

    # add expects float32
    try:
        index.add(vecs.astype(np.float32))
    except Exception as e:
        print(f"Warning: failed to add vectors to FAISS index: {e}")
        return None

    return index


def ann_attack_score(query_vecs: np.ndarray, index, k: int = 7, fallback_bank: np.ndarray = None) -> np.ndarray:
    """
    Compute mean top-k similarity using FAISS ANN.
    If FAISS unavailable or index is None, falls back to NumPy cosine + partial top-k.
    Returns higher-better similarity scores.
    """
    if fallback_bank is None:
        fallback_bank = np.empty((0, query_vecs.shape[1]), dtype=np.float32)

    if query_vecs is None or query_vecs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    # FAISS path
    if _FAISS_AVAILABLE and index is not None:
        Q = _normalize_rows_l2(query_vecs)
        k_eff = min(k, max(1, index.ntotal))
        try:
            D, _ = index.search(Q.astype(np.float32), k_eff)  # Inner product (similarity)
            # D shape: (nq, k_eff) similarity values
            return D.mean(axis=1).astype(np.float32)
        except Exception as e:
            print(f"Warning: FAISS search failed: {e}. Falling back to numpy.")
            # fall through to numpy fallback

    # Fallback: NumPy cosine similarity with partial top-k
    if fallback_bank is None or fallback_bank.shape[0] == 0:
        return np.zeros(query_vecs.shape[0], dtype=np.float32)
    Q = _normalize_rows_l2(query_vecs)
    B = _normalize_rows_l2(fallback_bank)
    sims = Q @ B.T  # (nq, nb)
    k_eff = min(k, sims.shape[1])
    if k_eff <= 0:
        return np.zeros(query_vecs.shape[0], dtype=np.float32)
    part = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
    return part.mean(axis=1).astype(np.float32)


# ----------------------- Threshold Finder -----------------------

def find_optimal_meta_threshold(y_true, y_scores, target_fpr=0.06):
    """
    Finds the best probability threshold that maximizes F1 under an FPR constraint,
    with a tie-breaker for the lowest FPR.
    """
    thresholds = np.unique(np.round(y_scores, 4))
    candidates = []
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tn, fp, fn, tp = _cm_2x2(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        if fpr <= target_fpr:
            f1 = f1_score(y_true, y_pred)
            candidates.append({"thresh": float(thresh), "f1": float(f1), "fpr": float(fpr)})

    if not candidates:
        print("Warning: No threshold met the FPR target. Falling back to max F1.")
        best_f1, best_thresh = -1.0, 0.5
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(thresh)
        return best_thresh

    max_f1 = max(c["f1"] for c in candidates)
    top_candidates = [c for c in candidates if c["f1"] >= max_f1 - 0.002]
    best_candidate = min(top_candidates, key=lambda x: x["fpr"])
    return best_candidate["thresh"]


# ----------------------------------------------------------------------------
# 5. MAIN EXECUTION BLOCK
# ----------------------------------------------------------------------------

def main():

    # === Configuration ===
    TRAIN_PATH = "../Datasets/UNSW/BUNSWTrain.csv"
    TEST_PATH = "../Datasets/UNSW/BUNSWTest.csv"

    SEED = 42
    FEATURE_DIM = 32
    TEMPERATURE = 0.38
    LR, EPOCHS, BATCH = 0.0018, 18, 128

    COVER_GAMMA = 0.97
    TARGET_FPR = 0.10  # Strict FPR target
    BGMM_CANDIDATES = [(3, 4e-3), (4, 8e-3), (5, 1e-2)]

    # >>> ANN CONFIG <<<
    ANN_METHOD = "flatip"  # "flatip" | "ivfpq" | "hnsw"
    ANN_K = 7
    ANN_NLIST = 256   # for ivfpq
    ANN_PQ_M = 16     # for ivfpq
    ANN_HNSW_M = 32   # for hnsw
    ANN_USE_GPU = False  # set True if faiss-gpu available
    # <<< ANN CONFIG >>>

    HCN_GUARD_QUANTILE = 0.995  # same as your original

    # --- Setup device selection ---
    if PREFERRED_TORCH_DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif PREFERRED_TORCH_DEVICE == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    setup_seed(SEED)


    # --- Load data ---
    X_train, y_train, X_test, y_test, df_test_orig, scaler, feature_cols, num_cols, cat_cols = load_and_preprocess_unsw_nb15(
        TRAIN_PATH, TEST_PATH
    )
    INPUT_DIM = X_train.shape[1]

    # ensure label tensors are integer dtype
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    # ----------------------------------------------------------------------
    # COMPUTATIONAL COST: start training timer (encoder + BGMM + meta)
    # ----------------------------------------------------------------------
    train_start = time.perf_counter()

    # --- Encoder Training ---
    model = AttentionFeatureExtractor(INPUT_DIM, FEATURE_DIM, dropout_rate=0.3).to(device)
    criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    print("\nSTARTING DEEP FEATURE EXTRACTOR TRAINING")
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        iters = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            z = model(xb)
            loss = criterion(z, yb)
            if not torch.isnan(loss):
                loss.backward()
                opt.step()
                total_loss += loss.item()
                iters += 1
        sch.step()
        avg_loss = (total_loss / iters) if iters > 0 else 0.0
        print(f"Epoch {ep}/{EPOCHS}, Avg Loss: {avg_loss:.6f}, LR: {sch.get_last_lr()[0]:.6f}")

    # --- Feature and Score Generation ---
    model.eval()
    with torch.no_grad():
        Z_list, y_list = [], []
        for xb, yb in DataLoader(train_ds, batch_size=BATCH, shuffle=False):
            z = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            Z_list.append(z)
            y_list.append(yb)
        Z_train_norm = torch.cat(Z_list, dim=0)
        y_tr = torch.cat(y_list, dim=0).numpy()

        z_norm_samples = Z_train_norm[y_tr == 0]
        if z_norm_samples.shape[0] >= 10:
            # distance to mean
            mean_vec = z_norm_samples.mean(dim=0, keepdim=True)
            cdist = 1.0 - F.cosine_similarity(z_norm_samples, mean_vec)
            keep_count = int(max(1, math.floor(0.9 * len(z_norm_samples))))
            idx_keep = torch.argsort(cdist)[:keep_count]
            z_bar = z_norm_samples[idx_keep].mean(dim=0)
        elif z_norm_samples.shape[0] > 0:
            z_bar = z_norm_samples.mean(dim=0)
        else:
            # fallback: mean of all train vectors
            z_bar = Z_train_norm.mean(dim=0)

        s_train = F.cosine_similarity(Z_train_norm, z_bar.unsqueeze(0)).clamp(-0.999999, 0.999999)
        z_train_scores = (0.5 * torch.log((1 + s_train) / (1 - s_train))).cpu().numpy()

    Z_train_np = Z_train_norm.numpy()

    # --- Calibration Phase ---
    print("\nSTARTING CALIBRATION PHASE")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, cal_idx = next(sss.split(np.arange(len(y_tr)), y_tr))

    z_tr_fit, y_tr_fit = z_train_scores[tr_idx], y_tr[tr_idx]
    z_tr_cal, y_tr_cal = z_train_scores[cal_idx], y_tr[cal_idx]
    Z_cal_raw = Z_train_np[cal_idx]  # NOTE: already L2-normalized from model step

    # BGMM Search on Calibration Set
    best_bgmm_config = {"f1": -1.0}
    for (NC, RC) in BGMM_CANDIDATES:
        try:
            bgmm_temp = fit_bgmm_on_scores(z_tr_fit, n_components=NC, seed=SEED, reg_covar=RC)
        except Exception as e:
            print(f"Warning: BGMM fit failed for n_components={NC}, reg_covar={RC}: {e}")
            continue
        comp_label_temp = map_components_to_labels_coverage(bgmm_temp, z_tr_fit, y_tr_fit, cover_gamma=COVER_GAMMA)
        pN_cal_raw_temp, _ = posterior_normal(bgmm_temp, z_tr_cal, comp_label_temp)

        # Using 1 - pN as a simple score for BGMM selection
        try:
            thresh_temp = find_optimal_meta_threshold(y_tr_cal, 1 - pN_cal_raw_temp, target_fpr=TARGET_FPR)
            y_pred_temp = (1 - pN_cal_raw_temp >= thresh_temp).astype(int)
            f1_temp = f1_score(y_tr_cal, y_pred_temp)
        except Exception:
            f1_temp = -1.0

        if f1_temp > best_bgmm_config["f1"]:
            best_bgmm_config = {
                "f1": f1_temp,
                "bgmm": bgmm_temp,
                "comp_label": comp_label_temp,
                "nc": NC,
                "rc": RC,
            }

    if "bgmm" not in best_bgmm_config:
        raise RuntimeError("Failed to fit any BGMM on calibration data.")
    bgmm = best_bgmm_config["bgmm"]
    comp_label = best_bgmm_config["comp_label"]
    print(f"[BGMM] Selected n_components={best_bgmm_config['nc']}, reg_covar={best_bgmm_config['rc']:.1e} via calibration search.")

    # --- Build Attack Bank & FAISS Index ---
    attack_bank = Z_cal_raw[(y_tr_cal == 1) & (z_tr_cal < -1.0)].astype(np.float32)
    if attack_bank.shape[0] > 1500:
        mbk = MiniBatchKMeans(n_clusters=200, random_state=SEED, n_init="auto")
        attack_bank = mbk.fit(attack_bank).cluster_centers_.astype(np.float32)

    faiss_index = build_faiss_index(
        attack_bank,
        method=ANN_METHOD,
        nlist=ANN_NLIST,
        pq_m=ANN_PQ_M,
        hnsw_m=ANN_HNSW_M,
        use_gpu=ANN_USE_GPU,
    )

    # --- Meta-Classifier Feature Engineering on Calibration Set ---
    pN_cal_raw, resp_cal = posterior_normal(bgmm, z_tr_cal, comp_label)

    ann_scores_cal = ann_attack_score(Z_cal_raw, faiss_index, k=ANN_K, fallback_bank=attack_bank)

    component_means = bgmm.means_[resp_cal.argmax(axis=1)].flatten()
    component_vars = bgmm.covariances_[resp_cal.argmax(axis=1)].flatten()
    meta_features_cal_list = [
        pN_cal_raw.reshape(-1, 1),
        ann_scores_cal.reshape(-1, 1),
        component_means.reshape(-1, 1),
        component_vars.reshape(-1, 1),
    ]
    meta_features_cal = np.hstack(meta_features_cal_list)

    # --- Meta-Classifier Training (Robust Logistic Regression) ---
    print("\nTRAINING ROBUST STACKED META-CLASSIFIER (Logistic Regression)")
    meta_pipe = Pipeline(
        [
            ("std", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0, class_weight="balanced", solver="liblinear", random_state=SEED
                ),
            ),
        ]
    )
    meta_pipe.fit(meta_features_cal, y_tr_cal)

    # --- Find Optimal Threshold and HCN Guard on Calibration Set ---
    meta_scores_cal = meta_pipe.predict_proba(meta_features_cal)[:, 1]
    optimal_threshold = find_optimal_meta_threshold(y_tr_cal, meta_scores_cal, target_fpr=TARGET_FPR)
    print(f"[Meta-Classifier] Optimal decision threshold found: {optimal_threshold:.4f}")

    try:
        hcn_guard_threshold = np.quantile(meta_scores_cal[y_tr_cal == 0], 1 - HCN_GUARD_QUANTILE)
    except Exception:
        hcn_guard_threshold = 0.0
    print(f"[HCN Guard] High-Confidence-Normal threshold set at: {hcn_guard_threshold:.4f}")

    # ----------------------------------------------------------------------
    # Stop training timer here (encoder + BGMM + FAISS + meta-classifier)
    # ----------------------------------------------------------------------
    train_end = time.perf_counter()
    training_time = train_end - train_start
    print(f"\n[Computational Cost] Training time T_train = {training_time:.2f} seconds")

    # --- Final Inference ---
    print("\nPERFORMING FINAL INFERENCE ON TEST SET")

    # start test timer (full pipeline inference on test set)
    test_start = time.perf_counter()

    y_pred_final, y_score_final = [], []
    for xb, _ in DataLoader(test_ds, batch_size=BATCH, shuffle=False):
        with torch.no_grad():
            z_batch_norm = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            s_batch = F.cosine_similarity(z_batch_norm, z_bar.unsqueeze(0)).clamp(-0.999999, 0.999999)
            z_batch_scores = (0.5 * torch.log((1 + s_batch) / (1 - s_batch))).cpu().numpy()

        pN_batch_raw, resp_batch = posterior_normal(bgmm, z_batch_scores, comp_label)

        ann_scores_batch = ann_attack_score(z_batch_norm.numpy(), faiss_index, k=ANN_K, fallback_bank=attack_bank)

        component_means_batch = bgmm.means_[resp_batch.argmax(axis=1)].flatten()
        component_vars_batch = bgmm.covariances_[resp_batch.argmax(axis=1)].flatten()

        meta_features_batch_list = [
            pN_batch_raw.reshape(-1, 1),
            ann_scores_batch.reshape(-1, 1),
            component_means_batch.reshape(-1, 1),
            component_vars_batch.reshape(-1, 1),
        ]
        meta_features_batch = np.hstack(meta_features_batch_list)

        y_scores = meta_pipe.predict_proba(meta_features_batch)[:, 1]
        y_pred = (y_scores >= optimal_threshold).astype(int)

        # Apply HCN Guard post-prediction
        y_pred[y_scores <= hcn_guard_threshold] = 0

        y_pred_final.append(y_pred)
        y_score_final.append(y_scores)

    y_pred = np.concatenate(y_pred_final)
    y_score = np.concatenate(y_score_final)

    # stop test timer
    test_end = time.perf_counter()
    test_time = test_end - test_start
    print(f"[Computational Cost] Test time T_test = {test_time:.2f} seconds for {len(y_test)} samples")

    # --- Metrics (printed + captured) ---
    acc, prec, rec, f1, fpr = score_detail(y_test, y_pred, y_score, title="Test Set Results (HAS-IDS)")

    # --- Save Predictions CSV ---
    pred_df = df_test_orig.copy()
    pred_df["binary_label"] = y_test
    pred_df["binary_pred"] = y_pred
    pred_df["binary_prob"] = y_score

    pred_path = "unsw_nb15_hasids_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # --- Save Metrics JSON ---
    try:
        roc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)
    except Exception:
        roc, pr_auc = None, None

    tn, fp, fn, tp = _cm_2x2(y_test, y_pred)
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(roc) if roc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "train_time_seconds": float(training_time),
        "test_time_seconds": float(test_time),
    }
    with open("unsw_nb15_hasids_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics + computational cost saved to unsw_nb15_hasids_test_metrics.json")

    # --- Save Artifacts ---
    print("\nSaving final HAS-IDS model artifacts...")
    try:
        artifacts = {
            "encoder_state_dict": model.state_dict(),
            "scaler": scaler,
            "bgmm": bgmm,
            "comp_label": comp_label,
            "z_bar": z_bar.cpu().numpy() if isinstance(z_bar, torch.Tensor) else np.array(z_bar),
            "meta_pipe": meta_pipe,
            "optimal_threshold": optimal_threshold,
            "hcn_guard_threshold": hcn_guard_threshold,
            "ann_config": {
                "method": ANN_METHOD,
                "k": ANN_K,
                "nlist": ANN_NLIST,
                "pq_m": ANN_PQ_M,
                "hnsw_m": ANN_HNSW_M,
                "use_gpu": ANN_USE_GPU,
            },
            "attack_bank": attack_bank,
            "feature_columns": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "model_version": "hasids_unsw_nb15",
        }
        joblib.dump(artifacts, "hasids_unsw_nb15_artifacts.joblib")
        print("Artifacts saved to hasids_unsw_nb15_artifacts.joblib")
    except Exception as e:
        print(f"Warning: failed to save artifacts: {e}")


if __name__ == "__main__":
    main()

