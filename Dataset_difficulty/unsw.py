#!/usr/bin/env python3

"""
UNSW-NB15 Difficulty / Score-Saturation Diagnostic
--------------------------------------------------

This is the UNSW-NB15 version of your OneR/ZeroR diagnostic, using:
- Train = BUNSWTrain.csv
- Test  = BUNSWTest.csv
- Feature subset:
  - Core numeric IDS features you listed
  - All one-hot proto_*, service_*, state_* columns

Pipeline:
- ZeroR baseline
- OneR per-feature analysis (DecisionTree depth=1, 1 feature at a time)
- Ensemble OneR (mean of useful depth-1 trees)
- Threshold via Youden's J on TRAIN
- AUROC / Precision / Recall / F1 on TEST
"""

import os
import pandas as pd
import numpy as np
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score,
    recall_score, f1_score, accuracy_score
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ##############################################################################
# --- 1. USER CONFIGURATION ---
# ##############################################################################

TRAIN_CSV = "../Datasets/UNSW/BUNSWTrain.csv"
TEST_CSV  = "../Datasets/UNSW/BUNSWTest.csv"

# default target name; will auto-switch to "Label" if needed
TARGET_COLUMN = "label"

# Base numeric features you explicitly care about (excluding 'label')
BASE_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
]

REPRODUCIBLE_RANDOM_STATE = 42

# ##############################################################################
# --- 2. HELPER FUNCTIONS ---
# ##############################################################################

def load_csv(path: str) -> pd.DataFrame:
    """Loads a single CSV and prints basic info."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df

def xs_y(df_: pd.DataFrame, targ: str, feature_cols):
    """
    Split DataFrame into features (xs) and target (y) using an explicit
    feature list.
    """
    xs = df_[feature_cols].copy()
    y = df_[targ].copy()
    return xs, y

def evaluate_one_feature(feature: str, metric=roc_auc_score):
    """
    Trains a OneR (DecisionTree max_depth=1) on a single feature.

    NOTE: This function relies on X_train, y_train, X_test, y_test
    existing in the GLOBAL SCOPE.
    """
    try:
        rootnode = DecisionTreeClassifier(
            max_depth=1,
            criterion='gini',
            random_state=REPRODUCIBLE_RANDOM_STATE
        )

        X_train_feature = X_train[[feature]].to_numpy()
        X_test_feature  = X_test[[feature]].to_numpy()

        rootnode.fit(X_train_feature, y_train)

        preds_test  = rootnode.predict_proba(X_test_feature)[:, 1]
        preds_train = rootnode.predict_proba(X_train_feature)[:, 1]

        met = round(metric(y_test, preds_test), 6)

        if met > 0.5:
            return [feature, met, rootnode, preds_test, preds_train]
        else:
            return [feature, met, None, None, None]

    except Exception:
        # feature crashes → treat as useless
        return [feature, 0.5, None, None, None]

# ##############################################################################
# --- 3. MAIN SCRIPT EXECUTION ---
# ##############################################################################

print("="*60)
print("--- 1. LOADING TRAIN & TEST ---")
print("="*60)

train_df = load_csv(TRAIN_CSV)
test_df  = load_csv(TEST_CSV)

# --- Ensure target column exists (handle 'label' vs 'Label') ---
if TARGET_COLUMN not in train_df.columns:
    if "Label" in train_df.columns:
        TARGET_COLUMN = "Label"
        print(f"\n[INFO] Using TARGET_COLUMN = '{TARGET_COLUMN}' instead.")
    else:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in train CSV "
            f"and no 'Label' column either."
        )

print("\n--- SANITY CHECK: Raw Label Counts (Train/Test) ---")
print("Train label counts:")
print(train_df[TARGET_COLUMN].value_counts())
print("\nTest label counts:")
print(test_df[TARGET_COLUMN].value_counts())
print("----------------------------------------------------------\n")

# --- If labels are strings (e.g. 'BENIGN' vs others), binarize like before ---
if train_df[TARGET_COLUMN].dtype == object:
    print("Binarizing textual labels: BENIGN -> 0, others -> 1")
    for df_ in (train_df, test_df):
        df_.loc[df_[TARGET_COLUMN] != 'BENIGN', TARGET_COLUMN] = 1
        df_.loc[df_[TARGET_COLUMN] == 'BENIGN', TARGET_COLUMN] = 0

# Force int32
train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(np.int32)
test_df[TARGET_COLUMN]  = test_df[TARGET_COLUMN].astype(np.int32)

print("Label distribution after binarization (Train):")
print(train_df[TARGET_COLUMN].value_counts(normalize=True))
print("\nLabel distribution after binarization (Test):")
print(test_df[TARGET_COLUMN].value_counts(normalize=True))

# --- 2. FEATURE SELECTION (your specified subset) ---
print("\n" + "="*60)
print("--- 2. FEATURE SUBSET SELECTION ---")
print("="*60)

all_train_cols = set(train_df.columns)

# 1) numeric base features that actually exist
base_feats_present = [f for f in BASE_FEATURES
                      if f in all_train_cols and f != TARGET_COLUMN]

# 2) all one-hot proto_*, service_*, state_* columns
proto_cols   = [c for c in all_train_cols if c.startswith("proto_")]
service_cols = [c for c in all_train_cols if c.startswith("service_")]
state_cols   = [c for c in all_train_cols if c.startswith("state_")]

feature_columns = sorted(set(base_feats_present
                             + proto_cols
                             + service_cols
                             + state_cols))

if not feature_columns:
    raise ValueError("No matching features found. Check your column names.")

print(f"Total number of selected features: {len(feature_columns)}")
print("Example features:", feature_columns[:15])

# --- 3. BUILD X/y FOR TRAIN & TEST ---
X_train, y_train = xs_y(train_df, targ=TARGET_COLUMN, feature_cols=feature_columns)
X_test,  y_test  = xs_y(test_df,  targ=TARGET_COLUMN, feature_cols=feature_columns)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# --- 4. Baseline 1: ZeroR (Majority Class) Model ---
print("\n" + "="*60)
print("--- 3. BASELINE 1: ZeroR (MAJORITY CLASS) ---")
print("="*60)

# predict all "0" (assumed majority benign)
zeror_preds = np.zeros_like(y_test)
zeror_acc   = accuracy_score(y_true=y_test, y_pred=zeror_preds)

try:
    zeror_auroc = roc_auc_score(y_true=y_test, y_score=zeror_preds)
except ValueError:
    # if only one class present, AUROC is undefined
    zeror_auroc = float("nan")

print(f"ZeroR Accuracy (predicting '0'): {zeror_acc:.4f}")
print(f"ZeroR AUROC (predicting '0'):   {zeror_auroc:.4f}")

# --- 5. Baseline 2: OneR (One-Rule) Analysis ---
print("\n" + "="*60)
print("--- 4. BASELINE 2: OneR (ONE-RULE) ANALYSIS ---")
print("="*60)
print(f"Evaluating {len(feature_columns)} features (sequential, Windows-safe)...")

# NO multiprocessing here – sequential loop
results = [evaluate_one_feature(feat) for feat in feature_columns]

result_df = pd.DataFrame(
    data=results,
    columns=["feature", "roc_auc_score", "model", "predictions", "preds_train"]
).sort_values(by="roc_auc_score", ascending=False)

print("\n--- OneR Top 15 Features (by AUROC on TEST) ---")
print(result_df[["feature", "roc_auc_score"]].head(15))

useful_features_df = result_df.loc[result_df["roc_auc_score"] > 0.5]
print(f"\n{len(useful_features_df)} / {len(feature_columns)} features have AUROC > 0.5.")

# --- 6. Baseline 3: Ensemble OneR Model ---
print("\n" + "="*60)
print("--- 5. BASELINE 3: ENSEMBLE OneR ---")
print("="*60)

if len(useful_features_df) > 0:
    preds_test_list  = [p for p in useful_features_df["predictions"].to_numpy() if p is not None]
    preds_train_list = [p for p in useful_features_df["preds_train"].to_numpy() if p is not None]

    if len(preds_test_list) == 0:
        raise ValueError("No valid predictions in useful_features_df; check your dataset.")

    ensemble_preds_test  = np.mean(np.vstack(preds_test_list), axis=0)
    ensemble_preds_train = np.mean(np.vstack(preds_train_list), axis=0)

    # Find threshold on TRAIN using Youden's J
    fpr, tpr, thresholds = roc_curve(y_train, ensemble_preds_train)
    J = tpr - fpr
    best_thresh_index = np.argmax(J)
    best_thresh = thresholds[best_thresh_index]

    print(f"Best threshold (found on TRAIN set): {best_thresh:.6f}")

    final_preds_binary = (ensemble_preds_test > best_thresh).astype(int)

    print("\n--- FINAL ENSEMBLE OneR METRICS (on TEST SET) ---")
    final_auroc = roc_auc_score(y_true=y_test, y_score=ensemble_preds_test)
    final_precision = precision_score(y_true=y_test, y_pred=final_preds_binary, zero_division=0)
    final_recall    = recall_score(y_true=y_test, y_pred=final_preds_binary, zero_division=0)
    final_f1        = f1_score(y_true=y_test, y_pred=final_preds_binary, zero_division=0)

    print(f"ROC-AUC:   {final_auroc:.6f}")
    print(f"Precision: {final_precision:.6f}")
    print(f"Recall:    {final_recall:.6f}")
    print(f"F1-Score:  {final_f1:.6f}")

    print("\n" + "="*60)
    print("--- 6. FINAL ANALYSIS & JUSTIFICATION ---")
    print("="*60)

    print("** CRITICAL FINDING **")
    print(f"Ensemble OneR AUROC on UNSW-NB15 (this split): {final_auroc:.6f}")
else:
    print("No useful (AUROC > 0.5) features were found. Something is off with this split/dataset.")
