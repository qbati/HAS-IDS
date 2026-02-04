import os
# ===== Quick switches (edit these) ===========================================
PREFERRED_TORCH_DEVICE = "cuda"   # "auto" | "cuda" | "cpu"
ANN_METHOD = "flatip"             # "flatip" | "ivfpq" | "hnsw"
ANN_K = 7
ANN_NLIST = 256   # for ivfpq
ANN_PQ_M = 16     # for ivfpq
ANN_HNSW_M = 32   # for hnsw
ANN_USE_GPU = False
# ============================================================================

# Cap CPU threading for reproducibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math, random, warnings, joblib, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

# Try to import FAISS
_FAISS_AVAILABLE = False
try:
    import faiss  # pip install faiss-cpu  (or faiss-gpu)
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


# ----------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------
def setup_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSeed {seed} set for reproducibility.")

def _cm_2x2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        pad = np.zeros((2, 2), dtype=int)
        pad[: cm.shape[0], : cm.shape[1]] = cm
        cm = pad
    return cm.ravel()

def score_detail(y_true, y_pred, y_score=None, title="Evaluation Results"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.size != 4: cm = np.zeros((2,2), dtype=int)
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
            pr  = average_precision_score(y_true, y_score)
            print(f"ROC-AUC:   {roc:.4f}")
            print(f"PR-AUC:    {pr:.4f}")
        except Exception:
            pass
    return acc, prec, rec, f1, fpr

def find_optimal_threshold(y_true, scores, target_fpr=0.06):
    """
    Given attack scores (higher = more attack), pick highest-F1 threshold
    under FPR <= target_fpr; tie-break by lowest FPR. If none, fall back to max F1.
    """
    if len(scores) == 0: return 0.5
    thresholds = np.unique(np.round(scores, 6))
    candidates = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tn, fp, fn, tp = _cm_2x2(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        if fpr <= target_fpr:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            candidates.append((t, f1, fpr))
    if not candidates:
        print("Warning: No threshold met FPR target. Falling back to max F1.")
        best_f1, best_t = -1.0, thresholds[len(thresholds)//2]
        for t in thresholds:
            f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        return float(best_t)
    max_f1 = max(c[1] for c in candidates)
    top = [c for c in candidates if c[1] >= max_f1 - 1e-3]
    best_t = min(top, key=lambda x: x[2])[0]
    return float(best_t)

def _normalize_rows_l2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def build_faiss_index(bank_vecs: np.ndarray,
                      method: str = "flatip",
                      nlist: int = 256, pq_m: int = 16, hnsw_m: int = 32,
                      use_gpu: bool = False):
    if bank_vecs is None or bank_vecs.shape[0] == 0: return None
    if not _FAISS_AVAILABLE: return None

    vecs = _normalize_rows_l2(bank_vecs)
    d = vecs.shape[1]

    if method == "flatip":
        index = faiss.IndexFlatIP(d)
    elif method == "ivfpq":
        quantizer = faiss.IndexFlatIP(d)
        nlist_eff = max(1, min(nlist, vecs.shape[0]))
        index = faiss.IndexIVFPQ(quantizer, d, nlist_eff, pq_m, 8)
        train_n = min(50000, len(vecs))
        train_sample = vecs[np.random.choice(len(vecs), train_n, replace=False)].astype(np.float32)
        try:
            index.train(train_sample)
        except Exception as e:
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
            print(f"Warning: failed to move FAISS index to GPU: {e}")

    try:
        index.add(vecs.astype(np.float32))
    except Exception as e:
        print(f"Warning: failed to add vectors to FAISS index: {e}")
        return None
    return index

def ann_attack_score(query_vecs: np.ndarray, index, k: int = 7, fallback_bank: np.ndarray = None) -> np.ndarray:
    """
    Mean top-k similarity to the attack bank. Higher = more attack-like.
    FAISS if available, else NumPy cosine fallback.
    """
    if query_vecs is None or query_vecs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if fallback_bank is None:
        fallback_bank = np.empty((0, query_vecs.shape[1]), dtype=np.float32)

    if _FAISS_AVAILABLE and index is not None:
        Q = _normalize_rows_l2(query_vecs)
        k_eff = min(k, max(1, index.ntotal))
        try:
            D, _ = index.search(Q.astype(np.float32), k_eff)  # similarities
            return D.mean(axis=1).astype(np.float32)
        except Exception as e:
            print(f"Warning: FAISS search failed: {e}. Falling back to numpy.")

    if fallback_bank.shape[0] == 0:
        return np.zeros(query_vecs.shape[0], dtype=np.float32)
    Q = _normalize_rows_l2(query_vecs)
    B = _normalize_rows_l2(fallback_bank)
    sims = Q @ B.T
    k_eff = min(k, sims.shape[1])
    if k_eff <= 0: return np.zeros(query_vecs.shape[0], dtype=np.float32)
    part = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
    return part.mean(axis=1).astype(np.float32)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_and_preprocess_unsw_nb15(train_path, test_path):
    df_train_orig = pd.read_csv(train_path)
    df_test_orig  = pd.read_csv(test_path)
    df_train, df_test = df_train_orig.copy(), df_test_orig.copy()
    for df in (df_train, df_test):
        if "id" in df.columns: df.drop(columns=["id"], inplace=True)

    y_train = df_train["label"].values.astype(int)
    y_test  = df_test["label"].values.astype(int)

    feature_cols = [c for c in df_train.columns if c not in ["label", "attack_cat"]]
    cat_cols = [c for c in feature_cols if df_train[c].dtype == "object"]
    num_cols = [c for c in feature_cols if df_train[c].dtype != "object"]

    X_train = pd.get_dummies(df_train[feature_cols], columns=cat_cols)
    X_test  = pd.get_dummies(df_test[feature_cols],  columns=cat_cols)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = MinMaxScaler()
    if len(num_cols) > 0:
        numeric_intersect = [c for c in num_cols if c in X_train.columns]
        if len(numeric_intersect) > 0:
            X_train[numeric_intersect] = scaler.fit_transform(X_train[numeric_intersect])
            X_test[numeric_intersect]  = scaler.transform(X_test[numeric_intersect])

    print(f"Preprocessing complete. Feature shape: {X_train.shape}")
    return (
        X_train.values.astype(np.float32), y_train,
        X_test.values.astype(np.float32),  y_test,
        df_test_orig, scaler, feature_cols, num_cols, cat_cols
    )


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
class AttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=32, dropout_rate=0.3):
        super().__init__()
        self.attn = nn.Linear(input_dim, input_dim)
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
        w = F.softmax(self.attn(x), dim=1)
        h = self.backbone(x * w)
        return self.head(h)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.38):
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
# MAIN — ANN-only ablation (Instance-based only; NO BGMM)
# ----------------------------------------------------------------------------
def main():
    # === Config ===
    TRAIN_PATH = "../Datasets/UNSW/BUNSWTrain.csv"
    TEST_PATH  = "../Datasets/UNSW/BUNSWTest.csv"

    SEED = 42
    FEATURE_DIM = 32
    LR, EPOCHS, BATCH = 0.0018, 18, 128
    TEMPERATURE = 0.38

    TARGET_FPR = 0.10
    HCN_GUARD_QUANTILE = 0.995   # very-normal protection (low ANN score)

    # Device
    if PREFERRED_TORCH_DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif PREFERRED_TORCH_DEVICE == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    setup_seed(SEED)

    # Data
    (X_train, y_train, X_test, y_test,
     df_test_orig, scaler, feature_cols, num_cols, cat_cols) = load_and_preprocess_unsw_nb15(
        TRAIN_PATH, TEST_PATH
    )
    INPUT_DIM = X_train.shape[1]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    # ----- Train encoder -----
    t0 = time.perf_counter()
    model = AttentionFeatureExtractor(INPUT_DIM, FEATURE_DIM, dropout_rate=0.3).to(device)
    criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    print("\nSTARTING DEEP FEATURE EXTRACTOR TRAINING (ANN-only)")
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss, iters = 0.0, 0
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
        avg = total_loss / max(1, iters)
        print(f"Epoch {ep}/{EPOCHS} | Avg Loss {avg:.6f} | LR {sch.get_last_lr()[0]:.6f}")

    # ----- Embed full train, build cal split -----
    model.eval()
    with torch.no_grad():
        Z_list, y_list = [], []
        for xb, yb in DataLoader(train_ds, batch_size=BATCH, shuffle=False):
            z = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            Z_list.append(z); y_list.append(yb)
        Z_train = torch.cat(Z_list, dim=0).numpy().astype(np.float32)
        y_tr = torch.cat(y_list, dim=0).numpy().astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, cal_idx = next(sss.split(np.arange(len(y_tr)), y_tr))
    Z_fit, y_fit = Z_train[tr_idx], y_tr[tr_idx]
    Z_cal, y_cal = Z_train[cal_idx], y_tr[cal_idx]

    # ----- Build ATTACK BANK from labeled attacks on D_cal -----
    attack_bank = Z_cal[y_cal == 1]
    if attack_bank.shape[0] == 0:
        # fall back: use most attack-like (furthest from mean normal) samples from Z_fit
        normals = Z_fit[y_fit == 0]
        if normals.shape[0] > 0:
            mu = normals.mean(axis=0, keepdims=True)
            sim = (normals @ mu.T).ravel()  # cosine since L2-normalized
            # pick a small set of lowest-similarity normals as pseudo-attacks (very conservative)
            k = max(50, int(0.01 * normals.shape[0]))
            idx = np.argsort(sim)[:k]
            attack_bank = normals[idx]
        else:
            # absolute worst-case: use all Z_cal
            attack_bank = Z_cal

    # (Optional) compress a huge bank (keeps speed/memory sane)
    if attack_bank.shape[0] > 5000:
        # simple random subsample; you can swap for MiniBatchKMeans cluster centers if you like
        sel = np.random.choice(attack_bank.shape[0], 5000, replace=False)
        attack_bank = attack_bank[sel]

    # Build FAISS index if possible
    faiss_index = build_faiss_index(
        attack_bank, method=ANN_METHOD, nlist=ANN_NLIST,
        pq_m=ANN_PQ_M, hnsw_m=ANN_HNSW_M, use_gpu=ANN_USE_GPU
    )

    # ----- Calibrate threshold on ANN scores (higher = more attack-like) -----
    ann_scores_cal = ann_attack_score(Z_cal, faiss_index, k=ANN_K, fallback_bank=attack_bank)
    optimal_threshold = find_optimal_threshold(y_cal, ann_scores_cal, target_fpr=TARGET_FPR)
    try:
        # very low ANN score among normals -> force NORMAL at inference
        hcn_guard_threshold = np.quantile(ann_scores_cal[y_cal == 0], 1 - HCN_GUARD_QUANTILE)
    except Exception:
        hcn_guard_threshold = float("-inf")

    t1 = time.perf_counter()
    print(f"[ANN-only] Optimal decision threshold = {optimal_threshold:.6f}")
    print(f"[HCN Guard] ann_score <= {hcn_guard_threshold:.6f} forced to NORMAL")
    print(f"\n[Computational Cost] Training time T_train = {t1 - t0:.2f} seconds")

    # ----- Inference on TEST -----
    print("\nPERFORMING FINAL INFERENCE ON TEST SET (ANN-only)")
    test_start = time.perf_counter()
    with torch.no_grad():
        Z_test_list, _ = [], []
        for xb, yb in DataLoader(test_ds, batch_size=BATCH, shuffle=False):
            z = F.normalize(model(xb.to(device)), p=2, dim=1).cpu().numpy().astype(np.float32)
            Z_test_list.append(z)
        Z_test = np.vstack(Z_test_list)

    ann_scores_test = ann_attack_score(Z_test, faiss_index, k=ANN_K, fallback_bank=attack_bank)
    y_scores = ann_scores_test.copy()
    y_pred = (y_scores >= optimal_threshold).astype(int)
    y_pred[y_scores <= hcn_guard_threshold] = 0

    test_end = time.perf_counter()
    print(f"[Computational Cost] Test time T_test = {test_end - test_start:.2f} seconds for {len(y_test)} samples")

    # ----- Metrics -----
    acc, prec, rec, f1, fpr = score_detail(
        y_test, y_pred, y_scores, title="Test Set Results (HAS-IDS Ablation: ANN-only)"
    )

    # ----- Save predictions -----
    pred_df = df_test_orig.copy()
    pred_df["binary_label"] = y_test
    pred_df["binary_pred"]  = y_pred
    pred_df["ann_score"]    = y_scores
    pred_path = "unsw_nb15_hasids_ann_only_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # ----- Save metrics -----
    try:
        roc = roc_auc_score(y_test, y_scores)
        pr  = average_precision_score(y_test, y_scores)
    except Exception:
        roc, pr = None, None

    tn, fp, fn, tp = _cm_2x2(y_test, y_pred)
    metrics = {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "fpr": float(fpr),
        "roc_auc": float(roc) if roc is not None else None,
        "pr_auc": float(pr) if pr is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "train_time_seconds": float(t1 - t0),
        "test_time_seconds": float(test_end - test_start),
    }
    with open("unsw_nb15_hasids_ann_only_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to unsw_nb15_hasids_ann_only_test_metrics.json")

    # ----- Save artifacts -----
    print("\nSaving HAS-IDS (ANN-only) artifacts...")
    try:
        artifacts = {
            "encoder_state_dict": model.state_dict(),
            "scaler": scaler,
            "attack_bank": attack_bank,  # store to rebuild index later
            "ann_config": {
                "method": ANN_METHOD, "k": ANN_K,
                "nlist": ANN_NLIST, "pq_m": ANN_PQ_M,
                "hnsw_m": ANN_HNSW_M, "use_gpu": ANN_USE_GPU
            },
            "optimal_threshold": optimal_threshold,
            "hcn_guard_threshold": hcn_guard_threshold,
            "feature_columns": feature_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "model_version": "hasids_unsw_nb15_ann_only",
        }
        joblib.dump(artifacts, "hasids_unsw_nb15_ann_only_artifacts.joblib")
        print("Artifacts saved to hasids_unsw_nb15_ann_only_artifacts.joblib")
    except Exception as e:
        print(f"Warning: failed to save artifacts: {e}")

if __name__ == "__main__":
    main()
