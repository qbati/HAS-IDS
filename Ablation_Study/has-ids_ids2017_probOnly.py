import os
# ===== Quick switches (edit these) ===========================================
PREFERRED_TORCH_DEVICE  = "cuda"   # "auto" | "cuda" | "cpu"
PREFERRED_FAISS_DEVICE  = "cpu"    # unused here, but kept for consistency
# ============================================================================

# (Optional) cap CPU threading to keep sklearn predictable
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import random
import warnings
import joblib
import json
import time

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
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings('ignore')


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


def _cm_2x2(y_true, y_pred):
    """Ensures confusion matrix is always 2x2 for stable metric calculation."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        pad = np.zeros((2, 2), dtype=int)
        pad[:cm.shape[0], :cm.shape[1]] = cm
        cm = pad
    return cm.ravel()


def score_detail(y_true, y_pred, y_score=None, title="Evaluation Results"):
    """Calculates and prints a comprehensive set of performance metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = _cm_2x2(y_true, y_pred)
    cm   = np.array([[tn, fp], [fn, tp]])
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

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


# ----------------------------------------------------------------------------
# 2. DATA LOADING AND PREPROCESSING (CIC-IDS-2017)
# ----------------------------------------------------------------------------

def load_and_preprocess_cicids2017(train_path, test_path):
    """
    Loads and preprocesses the CIC-IDS-2017 dataset.
    Handles infinite values and assumes all-numeric features.
    """
    print("Loading and preprocessing CIC-IDS-2017...")
    df_train_orig = pd.read_csv(train_path)
    df_test_orig  = pd.read_csv(test_path)

    df_train, df_test = df_train_orig.copy(), df_test_orig.copy()

    # Clean inf and NaN
    for df in (df_train, df_test):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    # The dataset is expected to be all-numeric except for label columns
    label_cols   = [c for c in ['label', 'attack_cat'] if c in df_train.columns]
    feature_cols = [c for c in df_train.columns if c not in label_cols]

    # Labels
    y_train = df_train['label'].values.astype(int)
    y_test  = df_test['label'].values.astype(int)

    # Features
    X_train = df_train[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()

    num_cols       = list(feature_cols)
    cat_cols       = []
    feature_cols_o = list(feature_cols)

    # Scale numeric
    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return (
        X_train.values.astype(np.float32), y_train,
        X_test.values.astype(np.float32),  y_test,
        df_test_orig, scaler,
        feature_cols_o, num_cols, cat_cols
    )


# ----------------------------------------------------------------------------
# 3. DEEP FEATURE EXTRACTOR AND LOSS
# ----------------------------------------------------------------------------

class AttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.encoder   = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        attn_scores  = self.attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        x_att        = x * attn_weights
        return self.encoder(x_att)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.4):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        z      = F.normalize(features, p=2, dim=1)
        labels = labels.view(-1, 1)
        mask   = torch.eq(labels, labels.T).float().to(device)

        logits      = (z @ z.T) / self.temperature
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask        = mask * logits_mask

        logits  = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_log = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_log.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob_pos.mean()


# ----------------------------------------------------------------------------
# 4. BGMM + Probabilistic Scoring (no context, no ANN)
# ----------------------------------------------------------------------------

def fit_bgmm_on_scores(z_scores_np, n_components=3, seed=42, reg_covar=4e-3):
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type='dirichlet_process',
        max_iter=500,
        n_init=1,
        random_state=seed,
        reg_covar=reg_covar
    )
    bgmm.fit(z_scores_np.reshape(-1, 1))
    return bgmm


def map_components_to_labels_coverage(bgmm, z_scores_np, y_np, cover_gamma=0.97):
    resp = bgmm.predict_proba(z_scores_np.reshape(-1, 1))
    lab  = np.array(['attack'] * bgmm.n_components, dtype=object)

    normal_idx = (y_np == 0)
    if normal_idx.sum() == 0:
        lab[np.argmax(bgmm.means_.flatten())] = 'normal'
        return lab

    mass_per_k  = resp[normal_idx].sum(axis=0)
    total_mass  = mass_per_k.sum()
    if total_mass <= 0:
        lab[np.argmax(bgmm.means_.flatten())] = 'normal'
        return lab

    order = np.argsort(bgmm.means_.flatten())[::-1]
    cum   = 0.0
    for k in order:
        lab[k] = 'normal'
        cum   += mass_per_k[k]
        if cum / total_mass >= cover_gamma:
            break
    return lab


def posterior_normal(bgmm, z_scores_np, comp_label):
    resp        = bgmm.predict_proba(z_scores_np.reshape(-1, 1))
    normal_mask = (comp_label == 'normal')
    if normal_mask.sum() == 0:
        return np.zeros(len(z_scores_np), dtype=np.float32), resp
    return resp[:, normal_mask].sum(axis=1), resp


def find_optimal_threshold(y_true, scores, target_fpr=0.06):
    """
    Generic threshold finder: maximize F1 under FPR constraint,
    then prefer lowest FPR among near-F1-optimal points.
    """
    thresholds = np.unique(np.round(scores, 4))
    candidates = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = _cm_2x2(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        if fpr <= target_fpr:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            candidates.append({'thr': float(thr), 'f1': float(f1), 'fpr': float(fpr)})

    if not candidates:
        print("Warning: No threshold met the FPR target. Falling back to max F1 on scores.")
        best_f1   = -1.0
        best_thr  = 0.5
        for thr in thresholds:
            y_pred = (scores >= thr).astype(int)
            f1     = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1  = float(f1)
                best_thr = float(thr)
        return best_thr

    max_f1 = max(c['f1'] for c in candidates)
    top    = [c for c in candidates if c['f1'] >= max_f1 - 0.002]
    best   = min(top, key=lambda c: c['fpr'])
    return float(best['thr'])


# ----------------------------------------------------------------------------
# 5. MAIN: HAS-IDS Prob-Only (no ANN, no context, no meta-classifier)
# ----------------------------------------------------------------------------

def main():
    # === Configuration for CIC-IDS-2017 ===
    TRAIN_PATH = "../Datasets/DCIC2017/DBcic2017_train.csv"
    TEST_PATH  = "../Datasets/DCIC2017/DBcic2017_test.csv"

    SEED        = 42
    FEATURE_DIM = 128
    TEMPERATURE = 0.05
    LR, EPOCHS, BATCH = 0.0020, 50, 256

    COVER_GAMMA = 0.97
    TARGET_FPR  = 0.10
    BGMM_CANDIDATES = [(3, 4e-3), (4, 8e-3), (5, 1e-2)]

    # No HCN guard / no meta-classifier in prob-only version

    # --- Device selection ---
    if PREFERRED_TORCH_DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif PREFERRED_TORCH_DEVICE == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    setup_seed(SEED)

    # --- Load data ---
    (X_train, y_train, X_test, y_test, df_test_orig, scaler,
     feature_columns, num_cols, cat_cols) = load_and_preprocess_cicids2017(TRAIN_PATH, TEST_PATH)

    INPUT_DIM = X_train.shape[1]
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    # ----------------------------------------------------------------------
    # TRAINING TIME: encoder + BGMM selection + threshold selection
    # ----------------------------------------------------------------------
    train_start = time.perf_counter()

    # --- Encoder training ---
    model = AttentionFeatureExtractor(INPUT_DIM, FEATURE_DIM, dropout_rate=0.3).to(device)
    criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    print("\nSTARTING DEEP FEATURE EXTRACTOR TRAINING (Prob-Only)")
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        iters      = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            z    = model(xb)
            loss = criterion(z, yb)
            if not torch.isnan(loss):
                loss.backward()
                opt.step()
                total_loss += loss.item()
                iters      += 1
        sch.step()
        avg_loss = (total_loss / iters) if iters > 0 else 0.0
        print(f"Epoch {ep}/{EPOCHS}, Avg Loss: {avg_loss:.6f}, LR: {sch.get_last_lr()[0]:.6f}")

    # --- Extract features & 1D scores for BGMM ---
    model.eval()
    with torch.no_grad():
        Z_list, y_list = [], []
        for xb, yb in DataLoader(train_ds, batch_size=512, shuffle=False):
            z = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            Z_list.append(z)
            y_list.append(yb)
        Z_train_norm = torch.cat(Z_list, dim=0)
        y_tr         = torch.cat(y_list, dim=0).numpy()

        z_norm_samples = Z_train_norm[y_tr == 0]
        if len(z_norm_samples) >= 10:
            cdist      = 1 - F.cosine_similarity(
                z_norm_samples, z_norm_samples.mean(dim=0, keepdim=True)
            )
            keep_count = int(max(1, math.floor(0.9 * len(z_norm_samples))))
            idx_keep   = torch.argsort(cdist)[:keep_count]
            z_bar      = z_norm_samples[idx_keep].mean(dim=0)
        else:
            z_bar = z_norm_samples.mean(dim=0)

        s_train        = F.cosine_similarity(Z_train_norm, z_bar.unsqueeze(0)).clamp(-1.0, 1.0)
        z_train_scores = (0.5 * torch.log((1 + s_train) / (1 - s_train + 1e-9))).cpu().numpy()

    # --- Calibration split ---
    Z_train_np = Z_train_norm.numpy()
    print("\nSTARTING CALIBRATION PHASE (Prob-Only)")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, cal_idx = next(sss.split(np.arange(len(y_tr)), y_tr))

    z_tr_fit, y_tr_fit = z_train_scores[tr_idx], y_tr[tr_idx]
    z_tr_cal, y_tr_cal = z_train_scores[cal_idx], y_tr[cal_idx]

    # --- BGMM + Threshold search on 1 - pN (attack score) ---
    best_cfg = {'f1': -1.0}
    for (NC, RC) in BGMM_CANDIDATES:
        bgmm_temp = fit_bgmm_on_scores(z_tr_fit, n_components=NC, seed=SEED, reg_covar=RC)
        comp_lab_temp = map_components_to_labels_coverage(
            bgmm_temp, z_tr_fit, y_tr_fit, cover_gamma=COVER_GAMMA
        )
        pN_cal_raw_temp, _ = posterior_normal(bgmm_temp, z_tr_cal, comp_lab_temp)

        # In prob-only variant, score = 1 - pN
        scores_temp = 1.0 - pN_cal_raw_temp
        thr_temp    = find_optimal_threshold(y_tr_cal, scores_temp, target_fpr=TARGET_FPR)
        y_pred_temp = (scores_temp >= thr_temp).astype(int)
        f1_temp     = f1_score(y_tr_cal, y_pred_temp, zero_division=0)

        if f1_temp > best_cfg['f1']:
            best_cfg = {
                'f1': float(f1_temp),
                'bgmm': bgmm_temp,
                'comp_label': comp_lab_temp,
                'nc': NC,
                'rc': RC,
                'thr': float(thr_temp),
            }

    bgmm        = best_cfg['bgmm']
    comp_label  = best_cfg['comp_label']
    best_thr    = best_cfg['thr']

    print(f"[BGMM Prob-Only] Selected n_components={best_cfg['nc']}, reg_covar={best_cfg['rc']:.1e}")
    print(f"[BGMM Prob-Only] Selected decision threshold (on 1 - pN): {best_thr:.4f}")

    # ----------------------------------------------------------------------
    # End of training time (encoder + BGMM + threshold)
    # ----------------------------------------------------------------------
    train_end      = time.perf_counter()
    training_time  = train_end - train_start
    print(f"\n[Computational Cost] Training time T_train = {training_time:.2f} seconds")

    # ----------------------------------------------------------------------
    # TEST / INFERENCE (Prob-only)
    # ----------------------------------------------------------------------
    print("\nPERFORMING FINAL INFERENCE ON TEST SET (HAS-IDS Prob-Only)")

    test_start = time.perf_counter()

    y_pred_final  = []
    y_score_final = []
    for xb, _ in DataLoader(test_ds, batch_size=512, shuffle=False):
        with torch.no_grad():
            z_batch_norm  = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            s_batch       = F.cosine_similarity(z_batch_norm, z_bar.unsqueeze(0)).clamp(-1.0, 1.0)
            z_batch_score = (0.5 * torch.log((1 + s_batch) / (1 - s_batch + 1e-9))).cpu().numpy()

        pN_batch_raw, _ = posterior_normal(bgmm, z_batch_score, comp_label)
        scores_batch    = 1.0 - pN_batch_raw  # attack probability-like score

        y_pred  = (scores_batch >= best_thr).astype(int)
        y_pred_final.append(y_pred)
        y_score_final.append(scores_batch)

    y_pred  = np.concatenate(y_pred_final)
    y_score = np.concatenate(y_score_final)

    test_end    = time.perf_counter()
    test_time   = test_end - test_start
    print(f"[Computational Cost] Test time T_test = {test_time:.2f} seconds for {len(y_test)} samples")

    # --- Metrics ---
    acc, prec, rec, f1, fpr = score_detail(
        y_test, y_pred, y_score,
        title="Test Set Results (HAS-IDS Prob-Only, CIC-IDS-2017)"
    )

    # --- Save predictions ---
    pred_df = df_test_orig.copy()
    pred_df["binary_label"] = y_test
    pred_df["binary_pred"]  = y_pred
    pred_df["binary_prob"]  = y_score  # 1 - pN score

    pred_path = "cicids2017_hasids_prob_only_0.001_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # --- Save metrics JSON ---
    try:
        roc    = roc_auc_score(y_test, y_score)
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
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "train_time_seconds": float(training_time),
        "test_time_seconds":  float(test_time),
    }
    with open("cicids2017_hasids_prob_only_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics + computational cost saved to cicids2017_hasids_prob_only_test_metrics.json")

    # --- Save artifacts ---
    print("\nSaving HAS-IDS Prob-Only artifacts for CIC-IDS-2017...")
    try:
        artifacts = {
            "encoder_state_dict": model.state_dict(),
            "scaler": scaler,
            "bgmm": bgmm,
            "comp_label": comp_label,
            "z_bar": z_bar.cpu().numpy(),
            "attack_bank": None,     # not used
            "optimal_threshold": best_thr,   # on 1 - pN
            "feature_columns": feature_columns,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "model_version": "hasids_prob_only_cicids2017_v1",
        }
        joblib.dump(artifacts, "cicids2017_hasids_prob_only_artifacts.joblib")
        print("Artifacts saved to cicids2017_hasids_prob_only_artifacts.joblib")
    except Exception as e:
        print(f"Warning: failed to save artifacts: {e}")


if __name__ == "__main__":
    main()
