import os
# ===== Quick switches (edit these) ===========================================
PREFERRED_TORCH_DEVICE  = "cuda"   # "auto" | "cuda" | "cpu"
# ============================================================================

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
# 1. UTILS
# ----------------------------------------------------------------------------

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"\nSeed {seed} set for reproducibility.")


def _cm_2x2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        pad = np.zeros((2, 2), dtype=int)
        pad[:cm.shape[0], :cm.shape[1]] = cm
        cm = pad
    return cm.ravel()


def score_detail(y_true, y_pred, y_score=None, title="Evaluation Results"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = _cm_2x2(y_true, y_pred)
    cm = np.array([[tn, fp], [fn, tp]])
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


def find_optimal_threshold(y_true, scores, target_fpr=0.06):
    scores = np.asarray(scores)
    if scores.size == 0:
        return 0.5

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
            y_pred = (scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t)

    max_f1 = max(c[1] for c in candidates)
    top = [c for c in candidates if c[1] >= max_f1 - 1e-3]
    best_t = min(top, key=lambda x: x[2])[0]
    return float(best_t)


# ----------------------------------------------------------------------------
# 2. DATA LOADER (DCICIDS2017)
# ----------------------------------------------------------------------------

def load_and_preprocess_dcicids2017(train_path, test_path):
    print("Loading and preprocessing DCIC-IDS-2017...")
    df_train_orig = pd.read_csv(train_path)
    df_test_orig  = pd.read_csv(test_path)

    df_train, df_test = df_train_orig.copy(), df_test_orig.copy()

    for df in (df_train, df_test):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    label_cols = [c for c in ['label', 'attack_cat'] if c in df_train.columns]
    feature_cols = [col for col in df_train.columns if col not in label_cols]

    y_train = df_train['label'].values.astype(int)
    y_test  = df_test['label'].values.astype(int)

    X_train = df_train[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()

    num_cols = list(feature_cols)
    cat_cols = []
    feature_columns = list(feature_cols)

    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return (
        X_train.values.astype(np.float32), y_train,
        X_test.values.astype(np.float32),  y_test,
        df_test_orig, scaler,
        feature_columns, num_cols, cat_cols
    )


# ----------------------------------------------------------------------------
# 3. ENCODER + SCL
# ----------------------------------------------------------------------------

class AttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        attention_scores = self.attention(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_features = x * attention_weights
        return self.encoder(attended_features)


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
# 4. BGMM CONTEXT
# ----------------------------------------------------------------------------

def fit_bgmm_on_scores(z_scores_np, n_components=3, seed=42, reg_covar=4e-3):
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type='dirichlet_process',
        max_iter=500, n_init=1, random_state=seed, reg_covar=reg_covar)
    bgmm.fit(z_scores_np.reshape(-1, 1))
    return bgmm


def map_components_to_labels_coverage(bgmm, z_scores_np, y_np, cover_gamma=0.97):
    resp = bgmm.predict_proba(z_scores_np.reshape(-1, 1))
    lab = np.array(['attack'] * bgmm.n_components, dtype=object)
    normal_idx = (y_np == 0)
    if normal_idx.sum() == 0:
        lab[np.argmax(bgmm.means_.flatten())] = 'normal'
        return lab
    mass_per_k = resp[normal_idx].sum(axis=0)
    total_mass = mass_per_k.sum()
    if total_mass <= 0:
        lab[np.argmax(bgmm.means_.flatten())] = 'normal'
        return lab
    order = np.argsort(bgmm.means_.flatten())[::-1]
    cum = 0.0
    for k in order:
        lab[k] = 'normal'
        cum += mass_per_k[k]
        if cum / total_mass >= cover_gamma:
            break
    return lab


def posterior_normal(bgmm, z_scores_np, comp_label):
    resp = bgmm.predict_proba(z_scores_np.reshape(-1, 1))
    normal_mask = (comp_label == 'normal')
    if normal_mask.sum() == 0:
        return np.zeros(len(z_scores_np), dtype=np.float32), resp
    return resp[:, normal_mask].sum(axis=1), resp


# ----------------------------------------------------------------------------
# 5. MAIN — CONTEXT ONLY (DCICIDS2017)
# ----------------------------------------------------------------------------

def main():
    TRAIN_PATH = "../Datasets/DCIC2017/DBcic2017_train.csv"
    TEST_PATH  = "../Datasets/DCIC2017/DBcic2017_test.csv"

    SEED = 42
    FEATURE_DIM = 128
    TEMPERATURE = 0.05
    LR, EPOCHS, BATCH = 0.0020, 50, 256

    COVER_GAMMA = 0.97
    TARGET_FPR = 0.10
    BGMM_CANDIDATES = [(3, 4e-3), (4, 8e-3), (5, 1e-2)]
    HCN_GUARD_QUANTILE = 0.995

    # device
    if PREFERRED_TORCH_DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif PREFERRED_TORCH_DEVICE == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    setup_seed(SEED)

    (X_train, y_train, X_test, y_test,
     df_test_orig, scaler,
     feature_columns, num_cols, cat_cols) = load_and_preprocess_dcicids2017(TRAIN_PATH, TEST_PATH)

    INPUT_DIM = X_train.shape[1]
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    # timer
    train_start = time.perf_counter()

    # encoder
    model = AttentionFeatureExtractor(INPUT_DIM, FEATURE_DIM, dropout_rate=0.3).to(device)
    criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    print("\nSTARTING DEEP FEATURE EXTRACTOR TRAINING (Context-only, DCIC)")
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

    # P-channel internal
    model.eval()
    with torch.no_grad():
        Z_list, y_list = [], []
        for xb, yb in DataLoader(train_ds, batch_size=512, shuffle=False):
            z = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            Z_list.append(z)
            y_list.append(yb)
        Z_train_norm = torch.cat(Z_list, dim=0)
        y_tr = torch.cat(y_list, dim=0).numpy()

        z_norm_samples = Z_train_norm[y_tr == 0]
        if len(z_norm_samples) >= 10:
            mean_vec = z_norm_samples.mean(dim=0, keepdim=True)
            cdist = 1.0 - F.cosine_similarity(z_norm_samples, mean_vec)
            keep_count = int(max(1, math.floor(0.9 * len(z_norm_samples))))
            idx_keep = torch.argsort(cdist)[:keep_count]
            z_bar = z_norm_samples[idx_keep].mean(dim=0)
        else:
            z_bar = z_norm_samples.mean(dim=0) if len(z_norm_samples) > 0 else Z_train_norm.mean(dim=0)

        s_train = F.cosine_similarity(Z_train_norm, z_bar.unsqueeze(0)).clamp(-0.999999, 0.999999)
        z_train_scores = (0.5 * torch.log((1 + s_train) / (1 - s_train))).cpu().numpy()

    # calibration
    print("\nSTARTING CALIBRATION PHASE (BGMM context, DCIC)")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, cal_idx = next(sss.split(np.arange(len(y_tr)), y_tr))

    z_tr_fit, y_tr_fit = z_train_scores[tr_idx], y_tr[tr_idx]
    z_tr_cal, y_tr_cal = z_train_scores[cal_idx], y_tr[cal_idx]

    best_cfg = {"f1": -1.0}
    for (NC, RC) in BGMM_CANDIDATES:
        try:
            bgmm_tmp = fit_bgmm_on_scores(z_tr_fit, n_components=NC, seed=SEED, reg_covar=RC)
        except Exception as e:
            print(f"Warning: BGMM fit failed for n_components={NC}, reg_covar={RC}: {e}")
            continue
        comp_label_tmp = map_components_to_labels_coverage(bgmm_tmp, z_tr_fit, y_tr_fit, cover_gamma=COVER_GAMMA)
        pN_cal_tmp, _ = posterior_normal(bgmm_tmp, z_tr_cal, comp_label_tmp)
        c_cal_tmp = 1.0 - pN_cal_tmp
        thresh_tmp = find_optimal_threshold(y_tr_cal, c_cal_tmp, target_fpr=TARGET_FPR)
        y_pred_tmp = (c_cal_tmp >= thresh_tmp).astype(int)
        f1_tmp = f1_score(y_tr_cal, y_pred_tmp, zero_division=0)

        if f1_tmp > best_cfg["f1"]:
            best_cfg = {
                "f1": f1_tmp,
                "bgmm": bgmm_tmp,
                "comp_label": comp_label_tmp,
                "nc": NC,
                "rc": RC,
                "thresh": thresh_tmp,
            }

    if "bgmm" not in best_cfg:
        raise RuntimeError("Failed to fit any BGMM on calibration data.")

    bgmm = best_cfg["bgmm"]
    comp_label = best_cfg["comp_label"]
    print(f"[BGMM] Selected n_components={best_cfg['nc']}, reg_covar={best_cfg['rc']:.1e} via calibration.")

    pN_cal, _ = posterior_normal(bgmm, z_tr_cal, comp_label)
    c_cal = 1.0 - pN_cal
    optimal_threshold = find_optimal_threshold(y_tr_cal, c_cal, target_fpr=TARGET_FPR)
    print(f"[Context-only DCIC] Optimal decision threshold (C) = {optimal_threshold:.6f}")

    try:
        hcn_guard_threshold = np.quantile(c_cal[y_tr_cal == 0], 1 - HCN_GUARD_QUANTILE)
    except Exception:
        hcn_guard_threshold = float("-inf")
    print(f"[HCN Guard] C <= {hcn_guard_threshold:.6f} forced to NORMAL")

    train_end = time.perf_counter()
    training_time = train_end - train_start
    print(f"\n[Computational Cost] Training time T_train = {training_time:.2f} seconds")

    # inference
    print("\nPERFORMING FINAL INFERENCE ON TEST SET (Context-only, DCIC)")
    test_start = time.perf_counter()

    y_scores_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, _ in DataLoader(test_ds, batch_size=512, shuffle=False):
            z_batch = F.normalize(model(xb.to(device)), p=2, dim=1).cpu()
            s_batch = F.cosine_similarity(z_batch, z_bar.unsqueeze(0)).clamp(-0.999999, 0.999999)
            z_scores_batch = (0.5 * torch.log((1 + s_batch) / (1 - s_batch))).cpu().numpy()

            pN_batch, _ = posterior_normal(bgmm, z_scores_batch, comp_label)
            c_batch = 1.0 - pN_batch

            y_scores = c_batch.copy()
            y_pred = (y_scores >= optimal_threshold).astype(int)
            y_pred[y_scores <= hcn_guard_threshold] = 0

            y_scores_list.append(y_scores)
            y_pred_list.append(y_pred)

    y_score = np.concatenate(y_scores_list)
    y_pred = np.concatenate(y_pred_list)

    test_end = time.perf_counter()
    test_time = test_end - test_start
    print(f"[Computational Cost] Test time T_test = {test_time:.2f} seconds for {len(y_test)} samples")

    # metrics
    acc, prec, rec, f1, fpr = score_detail(
        y_test, y_pred, y_score,
        title="Test Set Results (HAS-IDS Ablation: Context-only, DCICIDS2017)"
    )

    # predictions
    pred_df = df_test_orig.copy()
    pred_df["binary_label"]   = y_test
    pred_df["binary_pred"]    = y_pred
    pred_df["context_score"]  = y_score

    pred_path = "dcicids2017_hasids_context_only_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # metrics json
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
    with open("dcicids2017_hasids_context_only_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to dcicids2017_hasids_context_only_test_metrics.json")

    # artifacts
    print("\nSaving HAS-IDS (Context-only, DCICIDS2017) artifacts...")
    try:
        artifacts = {
            "encoder_state_dict": model.state_dict(),
            "scaler": scaler,
            "bgmm": bgmm,
            "comp_label": comp_label,
            "z_bar": z_bar.cpu().numpy() if isinstance(z_bar, torch.Tensor) else np.array(z_bar),
            "optimal_threshold": optimal_threshold,
            "hcn_guard_threshold": hcn_guard_threshold,
            "feature_columns": feature_columns,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "model_version": "hasids_dcicids2017_context_only",
        }
        joblib.dump(artifacts, "dcicids2017_hasids_context_only_artifacts.joblib")
        print("Artifacts saved to dcicids2017_hasids_context_only_artifacts.joblib")
    except Exception as e:
        print(f"Warning: failed to save artifacts: {e}")


if __name__ == "__main__":
    main()
