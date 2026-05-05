#!/usr/bin/env python
"""
run_10seeds_unsw.py  —  HAS-IDS reproducibility script
=======================================================
Runs the full HAS-IDS pipeline 10 times with different random seeds
on UNSW-NB15 and reports mean ± std for all metrics (Tables 6 & 8).

Usage:
    Place this file in the HAS-IDS/ folder, then run:
        python run_10seeds_unsw.py

    Edit the CONFIGURE section below to set dataset paths.

Output:
    unsw_10seeds_raw_results.csv  — per-seed raw metrics
    unsw_10seeds_summary.txt      — mean ± std + LaTeX-ready values

Note: has_ids_unsw.py is NOT modified by this script.
"""

# ============================================================
#  CONFIGURE ONLY THESE
# ============================================================
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]   # 10 seeds; 42 = original reported run

TRAIN_PATH           = r"../Datasets/UNSW/BUNSWTrain.csv"   # update if needed
TEST_PATH            = r"../Datasets/UNSW/BUNSWTest.csv"    # update if needed
MULTICLASS_TEST_PATH = r"../Datasets/UNSW/MUNSWTest.csv"    # multiclass labels

OUTPUT_RAW_CSV  = "unsw_10seeds_raw_results.csv"
OUTPUT_SUMMARY  = "unsw_10seeds_summary.txt"

PREFERRED_TORCH_DEVICE = "cuda"   # "cuda" | "cpu"
# ============================================================


# ---- cap CPU threading (matches has_ids_unsw.py) -----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import random
import warnings
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

_FAISS_AVAILABLE = False
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    pass


# ============================================================
#  UTILITY FUNCTIONS  (verbatim from has_ids_unsw.py — not changed)
# ============================================================

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"  [seed] {seed}")


def _cm_2x2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        pad = np.zeros((2, 2), dtype=int)
        pad[:cm.shape[0], :cm.shape[1]] = cm
        cm = pad
    return cm.ravel()


def load_and_preprocess_unsw_nb15(train_path, test_path):
    df_train_orig = pd.read_csv(train_path)
    df_test_orig  = pd.read_csv(test_path)
    df_train, df_test = df_train_orig.copy(), df_test_orig.copy()
    for df in (df_train, df_test):
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    y_train = df_train["label"].values.astype(int)
    y_test  = df_test["label"].values.astype(int)

    feature_cols = [c for c in df_train.columns if c not in ["label", "attack_cat"]]
    cat_cols = [c for c in feature_cols if df_train[c].dtype == "object"]
    num_cols = [c for c in feature_cols if df_train[c].dtype != "object"]

    X_train = pd.get_dummies(df_train[feature_cols], columns=cat_cols)
    X_test  = pd.get_dummies(df_test[feature_cols],  columns=cat_cols)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = MinMaxScaler()
    numeric_intersect = [c for c in num_cols if c in X_train.columns]
    if numeric_intersect:
        X_train[numeric_intersect] = scaler.fit_transform(X_train[numeric_intersect])
        X_test[numeric_intersect]  = scaler.transform(X_test[numeric_intersect])

    print(f"  Preprocessing done. Feature shape: {X_train.shape}")
    return (
        X_train.values.astype(np.float32),
        y_train,
        X_test.values.astype(np.float32),
        y_test,
        df_test_orig,
        scaler, feature_cols, num_cols, cat_cols,
    )


class AttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=32, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_rate),
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
        logits = (z @ z.T) / self.temperature                                    # original order
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob_pos.mean()


def fit_bgmm_on_scores(z_scores_np, n_components=3, seed=42, reg_covar=4e-3):
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type="dirichlet_process",
        max_iter=500, n_init=1, random_state=seed, reg_covar=reg_covar,
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
    total_mass  = mass_per_k.sum()
    if total_mass <= 0:
        lab[np.argmax(bgmm.means_.flatten())] = "normal"
        return lab
    order = np.argsort(bgmm.means_.flatten())[::-1]
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


def _normalize_rows_l2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_faiss_index(bank_vecs, method="flatip", nlist=256, pq_m=16, hnsw_m=32, use_gpu=False):
    if bank_vecs is None or bank_vecs.shape[0] == 0 or not _FAISS_AVAILABLE:
        return None
    vecs = _normalize_rows_l2(bank_vecs)
    d = vecs.shape[1]
    if method == "flatip":
        index = faiss.IndexFlatIP(d)
    elif method == "ivfpq":
        quantizer = faiss.IndexFlatIP(d)
        nlist_eff = max(1, min(nlist, vecs.shape[0]))
        index = faiss.IndexIVFPQ(quantizer, d, nlist_eff, pq_m, 8)
        train_n = min(50000, len(vecs))
        if train_n > 0:
            try:
                index.train(vecs[np.random.choice(len(vecs), train_n, replace=False)].astype(np.float32))
            except Exception as e:
                print(f"  IVFPQ train fallback: {e}")
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
            print(f"  FAISS GPU fallback: {e}")
    try:
        index.add(vecs.astype(np.float32))
    except Exception as e:
        print(f"  FAISS add failed: {e}")
        return None
    return index


def ann_attack_score(query_vecs, index, k=7, fallback_bank=None):
    if fallback_bank is None:
        fallback_bank = np.empty((0, query_vecs.shape[1]), dtype=np.float32)
    if query_vecs is None or query_vecs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if _FAISS_AVAILABLE and index is not None:
        Q = _normalize_rows_l2(query_vecs)
        k_eff = min(k, max(1, index.ntotal))
        try:
            D, _ = index.search(Q.astype(np.float32), k_eff)
            return D.mean(axis=1).astype(np.float32)
        except Exception as e:
            print(f"  FAISS search fallback: {e}")
    if fallback_bank.shape[0] == 0:
        return np.zeros(query_vecs.shape[0], dtype=np.float32)
    Q = _normalize_rows_l2(query_vecs)
    B = _normalize_rows_l2(fallback_bank)
    sims = Q @ B.T
    k_eff = min(k, sims.shape[1])
    if k_eff <= 0:
        return np.zeros(query_vecs.shape[0], dtype=np.float32)
    part = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
    return part.mean(axis=1).astype(np.float32)


def find_optimal_meta_threshold(y_true, y_scores, target_fpr=0.06):
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
        best_f1, best_thresh = -1.0, 0.5
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(thresh)
        return best_thresh
    max_f1 = max(c["f1"] for c in candidates)
    top_candidates = [c for c in candidates if c["f1"] >= max_f1 - 0.002]
    return min(top_candidates, key=lambda x: x["fpr"])["thresh"]


# ============================================================
#  PER-CLASS RECALL  (same logic as your per-class eval script)
# ============================================================

def compute_per_class_recall(y_pred, y_test, multi_df):
    """
    Merge binary predictions with multiclass labels and compute
    per-class recall (specificity for Normal, recall for attacks).
    Computes specificity for Normal class and recall for all attack classes.
    """
    results_df = multi_df[["attack_cat"]].copy().reset_index(drop=True)
    results_df["binary_label"] = y_test
    results_df["binary_pred"]  = y_pred
    results_df["attack_cat"]   = results_df["attack_cat"].astype(str).str.strip()

    recall_per_category = {}
    for category in sorted(results_df["attack_cat"].unique()):
        sub = results_df[results_df["attack_cat"] == category]
        true_s = sub["binary_label"]
        pred_s = sub["binary_pred"]
        if category.lower() == "normal":
            tn    = ((true_s == 0) & (pred_s == 0)).sum()
            total = len(sub)
            recall_per_category[category] = float(tn / total) if total > 0 else 0.0
        else:
            recall_per_category[category] = float(recall_score(true_s, pred_s, zero_division=0))
    return recall_per_category


# ============================================================
#  SINGLE SEED RUN  (method identical to has_ids_unsw.py main())
# ============================================================

def run_single_seed(seed, X_train, y_train, X_test, y_test, INPUT_DIM, device):
    """
    Full HAS-IDS pipeline for one seed.
    Method is UNCHANGED from has_ids_unsw.py — only the seed is parameterised.
    Returns:
        overall_metrics : dict  (raw fractions, not %)
        y_pred          : np.ndarray
        y_score         : np.ndarray
    """
    # ---- hyperparameters (has_ids_unsw.py) ----
    FEATURE_DIM        = 32
    TEMPERATURE        = 0.38
    LR, EPOCHS, BATCH  = 0.0018, 18, 128
    COVER_GAMMA        = 0.97
    TARGET_FPR         = 0.10
    BGMM_CANDIDATES    = [(3, 4e-3), (4, 8e-3), (5, 1e-2)]
    ANN_METHOD         = "flatip"
    ANN_K              = 7
    ANN_NLIST          = 256
    ANN_PQ_M           = 16
    ANN_HNSW_M         = 32
    ANN_USE_GPU        = False
    HCN_GUARD_QUANTILE = 0.995

    setup_seed(seed)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    # ---- encoder training ----
    model     = AttentionFeatureExtractor(INPUT_DIM, FEATURE_DIM, dropout_rate=0.3).to(device)
    criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)
    opt       = optim.Adam(model.parameters(), lr=LR)
    sch       = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    print("  Training encoder...")
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss, iters = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            z = model(xb)
            loss = criterion(z, yb)
            if not torch.isnan(loss):
                loss.backward(); opt.step()
                total_loss += loss.item(); iters += 1
        sch.step()
        avg = (total_loss / iters) if iters > 0 else 0.0
        if ep % 6 == 0 or ep == EPOCHS:
            print(f"    Epoch {ep:02d}/{EPOCHS}  loss={avg:.5f}  lr={sch.get_last_lr()[0]:.6f}")

    # ---- feature + score generation ----
    model.eval()
    with torch.no_grad():
        Z_list, y_list = [], []
        for xb, yb in DataLoader(train_ds, batch_size=BATCH, shuffle=False):
            z = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            Z_list.append(z); y_list.append(yb)
        Z_train_norm = torch.cat(Z_list, dim=0)
        y_tr         = torch.cat(y_list, dim=0).numpy()

        z_norm_samples = Z_train_norm[y_tr == 0]
        if z_norm_samples.shape[0] >= 10:
            mean_vec   = z_norm_samples.mean(dim=0, keepdim=True)
            cdist      = 1.0 - F.cosine_similarity(z_norm_samples, mean_vec)
            keep_count = int(max(1, math.floor(0.9 * len(z_norm_samples))))
            idx_keep   = torch.argsort(cdist)[:keep_count]
            z_bar      = z_norm_samples[idx_keep].mean(dim=0)
        elif z_norm_samples.shape[0] > 0:
            z_bar = z_norm_samples.mean(dim=0)
        else:
            z_bar = Z_train_norm.mean(dim=0)

        s_train         = F.cosine_similarity(Z_train_norm, z_bar.unsqueeze(0)).clamp(-0.999999, 0.999999)
        z_train_scores  = (0.5 * torch.log((1 + s_train) / (1 - s_train))).cpu().numpy()

    Z_train_np = Z_train_norm.numpy()

    # ---- calibration split ----
    print("  Calibration phase...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, cal_idx = next(sss.split(np.arange(len(y_tr)), y_tr))
    z_tr_fit, y_tr_fit = z_train_scores[tr_idx], y_tr[tr_idx]
    z_tr_cal, y_tr_cal = z_train_scores[cal_idx], y_tr[cal_idx]
    Z_cal_raw          = Z_train_np[cal_idx]

    # ---- BGMM search ----
    best_bgmm_config = {"f1": -1.0}
    for (NC, RC) in BGMM_CANDIDATES:
        try:
            bgmm_temp = fit_bgmm_on_scores(z_tr_fit, n_components=NC, seed=seed, reg_covar=RC)
        except Exception as e:
            print(f"  BGMM fit failed n={NC} rc={RC}: {e}"); continue
        comp_label_temp    = map_components_to_labels_coverage(bgmm_temp, z_tr_fit, y_tr_fit, cover_gamma=COVER_GAMMA)
        pN_cal_raw_temp, _ = posterior_normal(bgmm_temp, z_tr_cal, comp_label_temp)
        try:
            thresh_temp  = find_optimal_meta_threshold(y_tr_cal, 1 - pN_cal_raw_temp, target_fpr=TARGET_FPR)
            f1_temp      = f1_score(y_tr_cal, (1 - pN_cal_raw_temp >= thresh_temp).astype(int))
        except Exception:
            f1_temp = -1.0
        if f1_temp > best_bgmm_config["f1"]:
            best_bgmm_config = {"f1": f1_temp, "bgmm": bgmm_temp, "comp_label": comp_label_temp, "nc": NC, "rc": RC}

    if "bgmm" not in best_bgmm_config:
        raise RuntimeError("Failed to fit any BGMM on calibration data.")
    bgmm       = best_bgmm_config["bgmm"]
    comp_label = best_bgmm_config["comp_label"]
    print(f"  BGMM selected: n={best_bgmm_config['nc']}, reg={best_bgmm_config['rc']:.1e}")

    # ---- attack bank + FAISS ----
    attack_bank = Z_cal_raw[(y_tr_cal == 1) & (z_tr_cal < -1.0)].astype(np.float32)
    if attack_bank.shape[0] > 1500:
        mbk         = MiniBatchKMeans(n_clusters=200, random_state=seed, n_init="auto")
        attack_bank = mbk.fit(attack_bank).cluster_centers_.astype(np.float32)

    faiss_index = build_faiss_index(attack_bank, method=ANN_METHOD, nlist=ANN_NLIST,
                                     pq_m=ANN_PQ_M, hnsw_m=ANN_HNSW_M, use_gpu=ANN_USE_GPU)

    # ---- meta-classifier features on calibration ----
    pN_cal_raw, resp_cal = posterior_normal(bgmm, z_tr_cal, comp_label)
    ann_scores_cal       = ann_attack_score(Z_cal_raw, faiss_index, k=ANN_K, fallback_bank=attack_bank)
    component_means      = bgmm.means_[resp_cal.argmax(axis=1)].flatten()
    component_vars       = bgmm.covariances_[resp_cal.argmax(axis=1)].flatten()
    meta_features_cal    = np.hstack([
        pN_cal_raw.reshape(-1, 1),
        ann_scores_cal.reshape(-1, 1),
        component_means.reshape(-1, 1),
        component_vars.reshape(-1, 1),
    ])

    # ---- meta-classifier training ----
    print("  Training meta-classifier...")
    meta_pipe = Pipeline([
        ("std", StandardScaler()),
        ("lr",  LogisticRegression(C=1.0, class_weight="balanced",
                                   solver="liblinear", random_state=seed)),
    ])
    meta_pipe.fit(meta_features_cal, y_tr_cal)

    meta_scores_cal  = meta_pipe.predict_proba(meta_features_cal)[:, 1]
    optimal_threshold = find_optimal_meta_threshold(y_tr_cal, meta_scores_cal, target_fpr=TARGET_FPR)
    try:
        hcn_guard_threshold = np.quantile(meta_scores_cal[y_tr_cal == 0], 1 - HCN_GUARD_QUANTILE)
    except Exception:
        hcn_guard_threshold = 0.0
    print(f"  Threshold={optimal_threshold:.4f}  HCN guard={hcn_guard_threshold:.4f}")

    # ---- inference on test set ----
    print("  Inferring on test set...")
    y_pred_final, y_score_final = [], []
    for xb, _ in DataLoader(test_ds, batch_size=BATCH, shuffle=False):
        with torch.no_grad():
            z_batch_norm = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            s_batch      = F.cosine_similarity(z_batch_norm, z_bar.unsqueeze(0)).clamp(-0.999999, 0.999999)
            z_batch_scores = (0.5 * torch.log((1 + s_batch) / (1 - s_batch))).cpu().numpy()

        pN_batch_raw, resp_batch = posterior_normal(bgmm, z_batch_scores, comp_label)
        ann_scores_batch         = ann_attack_score(z_batch_norm.numpy(), faiss_index, k=ANN_K, fallback_bank=attack_bank)
        comp_means_b  = bgmm.means_[resp_batch.argmax(axis=1)].flatten()
        comp_vars_b   = bgmm.covariances_[resp_batch.argmax(axis=1)].flatten()
        meta_feat_b   = np.hstack([
            pN_batch_raw.reshape(-1, 1),
            ann_scores_batch.reshape(-1, 1),
            comp_means_b.reshape(-1, 1),
            comp_vars_b.reshape(-1, 1),
        ])

        y_scores = meta_pipe.predict_proba(meta_feat_b)[:, 1]
        y_pred   = (y_scores >= optimal_threshold).astype(int)
        y_pred[y_scores <= hcn_guard_threshold] = 0

        y_pred_final.append(y_pred)
        y_score_final.append(y_scores)

    y_pred  = np.concatenate(y_pred_final)
    y_score = np.concatenate(y_score_final)

    # ---- overall metrics ----
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = _cm_2x2(y_test, y_pred)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc  = average_precision_score(y_test, y_score)
    except Exception:
        roc_auc = pr_auc = None

    overall_metrics = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "fpr":       fpr,
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
    }
    return overall_metrics, y_pred, y_score


# ============================================================
#  SUMMARY + OUTPUT
# ============================================================

OVERALL_KEYS   = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]
OVERALL_LABELS = ["Acc",      "Pre",       "Rec",    "F1", "FPR", "ROC-AUC", "PR-AUC"]


def _fmt(val, std=None):
    if val is None:
        return "N/A"
    s = f"{val:.2f}"
    if std is not None:
        s += f" ± {std:.2f}"
    return s


def build_summary(all_overall, all_perclass, seeds):
    lines = []

    # ---- per-seed raw table ----
    lines.append("=" * 80)
    lines.append("PER-SEED OVERALL PERFORMANCE  (%)")
    lines.append("=" * 80)
    hdr = f"{'Metric':<12}" + "".join(f"{'S'+str(s):>9}" for s in seeds)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for key, lbl in zip(OVERALL_KEYS, OVERALL_LABELS):
        vals = [o[key] * 100 if o[key] is not None else float("nan") for o in all_overall]
        row  = f"{lbl:<12}" + "".join(f"{v:>9.2f}" for v in vals)
        lines.append(row)

    lines.append("")
    lines.append("PER-SEED PER-CLASS RECALL  (%)")
    lines.append("-" * 80)
    all_cats = sorted(all_perclass[0].keys(), key=lambda x: (x.lower() != "normal", x.lower()))
    hdr2 = f"{'Class':<28}" + "".join(f"{'S'+str(s):>9}" for s in seeds)
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    for cat in all_cats:
        vals = [pc[cat] * 100 for pc in all_perclass]
        row  = f"{cat:<28}" + "".join(f"{v:>9.2f}" for v in vals)
        lines.append(row)

    # ---- mean ± std tables ----
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"MEAN ± STD  (across {len(seeds)} seeds)")
    lines.append("=" * 80)

    lines.append("\n--- OVERALL PERFORMANCE ---")
    lines.append(f"{'Metric':<12}  {'Mean ± Std':>14}")
    lines.append("-" * 30)
    for key, lbl in zip(OVERALL_KEYS, OVERALL_LABELS):
        vals = [o[key] * 100 if o[key] is not None else float("nan") for o in all_overall]
        m, s = np.nanmean(vals), np.nanstd(vals)
        lines.append(f"  {lbl:<10}  {m:>6.2f} ± {s:.2f}")

    lines.append("\n--- PER-CLASS RECALL ---")
    lines.append(f"{'Class':<28}  {'Mean ± Std':>14}")
    lines.append("-" * 46)
    for cat in all_cats:
        vals = [pc[cat] * 100 for pc in all_perclass]
        m, s = np.mean(vals), np.std(vals)
        lines.append(f"  {cat:<26}  {m:>6.2f} ± {s:.2f}")

    # ---- LaTeX-ready block ----
    lines.append("")
    lines.append("=" * 80)
    lines.append("LATEX-READY VALUES  (paste into Step 25)")
    lines.append("=" * 80)

    lines.append("\n-- Table 6 (UNSW Overall, HAS-IDS row) --")
    parts = []
    for key, lbl in zip(OVERALL_KEYS, OVERALL_LABELS):
        vals = [o[key] * 100 if o[key] is not None else float("nan") for o in all_overall]
        m, s = np.nanmean(vals), np.nanstd(vals)
        parts.append(f"{lbl}: {m:.2f} ± {s:.2f}")
    lines.append("  " + " | ".join(parts))

    lines.append("\n-- Table 8 (UNSW Per-Class, HAS-IDS column) --")
    for cat in all_cats:
        vals = [pc[cat] * 100 for pc in all_perclass]
        m, s = np.mean(vals), np.std(vals)
        lines.append(f"  {cat:<28}: {m:.2f} ± {s:.2f}")

    return "\n".join(lines)


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" HAS-IDS UNSW-NB15 — 10-SEED EVALUATION")
    print(f" Seeds: {SEEDS}")
    print("=" * 60)

    # ---- device ----
    if PREFERRED_TORCH_DEVICE == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # ---- load data ONCE (preprocessing is deterministic) ----
    print("\n[DATA] Loading and preprocessing...")
    (X_train, y_train, X_test, y_test,
     df_test_orig, scaler, feature_cols, num_cols, cat_cols) = load_and_preprocess_unsw_nb15(TRAIN_PATH, TEST_PATH)
    INPUT_DIM = X_train.shape[1]

    # ---- load multiclass labels ONCE ----
    print(f"[DATA] Loading multiclass labels: {MULTICLASS_TEST_PATH}")
    multi_df = pd.read_csv(MULTICLASS_TEST_PATH)
    if len(multi_df) != len(y_test):
        raise ValueError(f"Row count mismatch: multiclass={len(multi_df)}, test={len(y_test)}")
    print(f"  Multiclass labels loaded. {len(multi_df)} rows. "
          f"Classes: {sorted(multi_df['attack_cat'].astype(str).str.strip().unique())}")

    # ---- run all seeds ----
    all_overall  = []
    all_perclass = []
    all_rows     = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'=' * 60}")
        print(f"  RUN {i+1}/{len(SEEDS)}   SEED = {seed}")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()

        overall, y_pred, y_score = run_single_seed(
            seed, X_train, y_train, X_test, y_test, INPUT_DIM, device
        )
        perclass = compute_per_class_recall(y_pred, y_test, multi_df)

        elapsed = time.perf_counter() - t0
        all_overall.append(overall)
        all_perclass.append(perclass)

        # build flat row for CSV
        row = {"seed": seed}
        for k in OVERALL_KEYS:
            row[k] = (overall[k] * 100) if overall[k] is not None else None
        for cat, val in perclass.items():
            row[f"recall_{cat}"] = val * 100
        all_rows.append(row)

        print(f"\n  ✓ Seed {seed} done in {elapsed/60:.1f} min  "
              f"F1={overall['f1']*100:.2f}%  FPR={overall['fpr']*100:.2f}%")

    # ---- save raw results CSV ----
    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(OUTPUT_RAW_CSV, index=False)
    print(f"\n[SAVED] Raw per-seed results → {OUTPUT_RAW_CSV}")

    # ---- build + print + save summary ----
    summary = build_summary(all_overall, all_perclass, SEEDS)
    print("\n" + summary)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n[SAVED] Summary → {OUTPUT_SUMMARY}")
    print("\nAll done. Copy the LATEX-READY VALUES block into Step 25.")


if __name__ == "__main__":
    main()