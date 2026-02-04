import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# --- Config ---
TRAIN_PATH = "../Datasets/DCIC2017/DBcic2017_train.csv"
TEST_PATH  = "../Datasets/DCIC2017/DBcic2017_test.csv"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE      = 512
EPOCHS          = 10
LR              = 1e-3
TAU             = 0.075
BETA            = 0.15
BGMM_COMPONENTS = 20
NORMAL_FRAC     = 0.80   # component >80% normal -> "normal"


# --- Encoder: 82 -> 64 -> 64 -> 32 ---
class EncoderDCIC(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)


# --- Asymmetric contrastive loss ---
def asymmetric_supcon_loss(z, y, tau=TAU, beta=BETA):
    normal_mask = (y == 0)
    attack_mask = (y == 1)

    if normal_mask.sum() < 2 or attack_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    v_n = F.normalize(z[normal_mask], p=2, dim=1)
    v_a = F.normalize(z[attack_mask], p=2, dim=1)

    sim_nn = v_n @ v_n.T
    sim_na = v_n @ v_a.T

    t_ij = torch.exp(sim_nn / tau)
    w_il = 1.0 / torch.exp(sim_nn / beta)

    sum_na = torch.exp(sim_na / tau).sum(dim=1)
    full_den = t_ij + torch.matmul(w_il, sum_na).unsqueeze(1)

    mask = ~torch.eye(v_n.size(0), dtype=bool, device=z.device)
    num = t_ij[mask]
    den = full_den[mask] + 1e-9

    return -torch.log(num / den).mean()


def main():
    print("====== DCIC2017 =====")
    print(f"Device: {DEVICE}")

    # 1) Load
    df_tr = pd.read_csv(TRAIN_PATH)
    df_te = pd.read_csv(TEST_PATH)

    # pull labels
    if "label" in df_tr.columns:
        y_tr = df_tr.pop("label").astype(int).values
        y_te = df_te.pop("label").astype(int).values
    else:
        y_tr = df_tr.iloc[:, -1].astype(int).values
        y_te = df_te.iloc[:, -1].astype(int).values
        df_tr = df_tr.iloc[:, :-1]
        df_te = df_te.iloc[:, :-1]

    # 2) Global preprocessing (numeric + global MinMax)
    print("Applying GLOBAL MinMax scaling (train + test)...")
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_all = df_all.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(df_all.values.astype(np.float32)).astype(np.float32)

    X_tr = X_all_scaled[: len(df_tr)]
    X_te = X_all_scaled[len(df_tr) :]

    if X_tr.shape[1] != 82:
        raise ValueError(f"Expected 82-dim features, got {X_tr.shape[1]}")

    # 3) Train encoder
    model = EncoderDCIC(X_tr.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    print("Training encoder...")
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            z = model(xb)
            loss = asymmetric_supcon_loss(z, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / max(len(train_loader), 1)
        print(f"  Epoch {ep+1}/{EPOCHS} - loss: {avg_loss:.4f}")
    T_train_enc = time.perf_counter() - t0

    # 4) Embeddings
    model.eval()
    with torch.no_grad():
        Z_tr = model(torch.from_numpy(X_tr).to(DEVICE)).cpu().numpy()
        Z_te = model(torch.from_numpy(X_te).to(DEVICE)).cpu().numpy()

    # 5) Cosine scores vs normal template
    norm_temp = Z_tr[y_tr == 0].mean(axis=0)

    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    tr_sc = np.array([cosine(z, norm_temp) for z in Z_tr], dtype=np.float32).reshape(-1, 1)
    te_sc = np.array([cosine(z, norm_temp) for z in Z_te], dtype=np.float32).reshape(-1, 1)

    # 6) BGMM
    print("Fitting BGMM...")
    t1 = time.perf_counter()
    bgmm = BayesianGaussianMixture(
        n_components=BGMM_COMPONENTS,
        weight_concentration_prior_type="dirichlet_process",
        max_iter=500
    ).fit(tr_sc)
    T_train_bgmm = time.perf_counter() - t1

    # 7) Component mapping
    tr_comp = bgmm.predict(tr_sc)
    comp_map = np.array(["attack"] * BGMM_COMPONENTS, dtype=object)
    for k in range(BGMM_COMPONENTS):
        idx = (tr_comp == k)
        if not idx.any():
            continue
        ratio_norm = (y_tr[idx] == 0).mean()
        if ratio_norm > NORMAL_FRAC:
            comp_map[k] = "normal"

    # 8) Inference
    t2 = time.perf_counter()
    te_comp = bgmm.predict(te_sc)
    y_pred = np.array(
        [0 if comp_map[c] == "normal" else 1 for c in te_comp],
        dtype=int
    )
    T_test = time.perf_counter() - t2

    # Probabilistic scores for ROC/PR
    post_te = bgmm.predict_proba(te_sc)
    attack_mask = np.array(
        [1 if comp_map[i] == "attack" else 0 for i in range(BGMM_COMPONENTS)],
        dtype=float
    )
    y_score = (post_te * attack_mask).sum(axis=1)

    # 9) Metrics
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    roc_auc = roc_auc_score(y_te, y_score)
    pr_auc  = average_precision_score(y_te, y_score)

    print("\n--- DCIC2017 Results ---")
    print("Confusion matrix:\n", cm)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"FPR      : {fpr:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Train time (enc+BGMM): {T_train_enc + T_train_bgmm:.2f}s")
    print(f"Test time            : {T_test:.4f}s")

    # 10) Save artifacts
    pd.DataFrame(
        {"y_true": y_te, "y_pred": y_pred, "y_score": y_score}
    ).to_csv("DCIC2017_preds.csv", index=False)
    joblib.dump(
        {"bgmm": bgmm, "comp_map": comp_map},
        "DCIC2017_bgmm.joblib"
    )


if __name__ == "__main__":
    main()
