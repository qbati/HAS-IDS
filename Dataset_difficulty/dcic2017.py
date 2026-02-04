#!/usr/bin/env python3

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

TRAIN_CSV =  "../Datasets/DCIC2017/DBcic2017_train.csv"
TEST_CSV  = "../Datasets/DCIC2017/DBcic2017_test.csv"

TARGET_COLUMN = "label"

BASE_FEATURES = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd RST Flags",
    "Bwd RST Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Avg Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Fwd Act Data Packets",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "ICMP Code",
    "ICMP Type",
    "Total TCP Flow Time"
]

REPRODUCIBLE_RANDOM_STATE = 42

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df

def xs_y(df_: pd.DataFrame, targ: str, feature_cols):
    xs = df_[feature_cols].copy()
    y = df_[targ].copy()
    return xs, y

def evaluate_one_feature(feature: str, metric=roc_auc_score):
    """
    OneR: depth-1 tree on a single feature.
    Uses global X_train, y_train, X_test, y_test.
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
        return [feature, 0.5, None, None, None]

print("="*60)
print("--- 1. LOADING TRAIN & TEST ---")
print("="*60)

train_df = load_csv(TRAIN_CSV)
test_df  = load_csv(TEST_CSV)

# handle 'label' vs 'Label'
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

# Binarize textual labels if needed
if train_df[TARGET_COLUMN].dtype == object:
    print("Binarizing textual labels: BENIGN -> 0, others -> 1")
    for df_ in (train_df, test_df):
        df_.loc[df_[TARGET_COLUMN] != 'BENIGN', TARGET_COLUMN] = 1
        df_.loc[df_[TARGET_COLUMN] == 'BENIGN', TARGET_COLUMN] = 0

train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(np.int32)
test_df[TARGET_COLUMN]  = test_df[TARGET_COLUMN].astype(np.int32)

print("Label distribution after binarization (Train):")
print(train_df[TARGET_COLUMN].value_counts(normalize=True))
print("\nLabel distribution after binarization (Test):")
print(test_df[TARGET_COLUMN].value_counts(normalize=True))

print("\n" + "="*60)
print("--- 2. FEATURE SUBSET SELECTION ---")
print("="*60)

all_train_cols = set(train_df.columns)

feature_columns = [f for f in BASE_FEATURES
                   if f in all_train_cols and f != TARGET_COLUMN]

if not feature_columns:
    raise ValueError("No matching features found. Check your column names.")

print(f"Total feature columns selected: {len(feature_columns)}")
print("Example features:", feature_columns[:15])

X_train, y_train = xs_y(train_df, targ=TARGET_COLUMN, feature_cols=feature_columns)
X_test,  y_test  = xs_y(test_df,  targ=TARGET_COLUMN, feature_cols=feature_columns)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

print("\n" + "="*60)
print("--- 3. BASELINE 1: ZeroR (MAJORITY CLASS) ---")
print("="*60)

zeror_preds = np.zeros_like(y_test)
zeror_acc   = accuracy_score(y_true=y_test, y_pred=zeror_preds)

try:
    zeror_auroc = roc_auc_score(y_true=y_test, y_score=zeror_preds)
except ValueError:
    zeror_auroc = float("nan")

print(f"ZeroR Accuracy (predicting '0'): {zeror_acc:.4f}")
print(f"ZeroR AUROC (predicting '0'):   {zeror_auroc:.4f}")

print("\n" + "="*60)
print("--- 4. BASELINE 2: OneR (ONE-RULE) ANALYSIS ---")
print("="*60)
print(f"Evaluating {len(feature_columns)} features (sequential, Windows-safe)...")

# NO multiprocessing – sequential loop
results = [evaluate_one_feature(feat) for feat in feature_columns]

result_df = pd.DataFrame(
    data=results,
    columns=["feature", "roc_auc_score", "model", "predictions", "preds_train"]
).sort_values(by="roc_auc_score", ascending=False)

print("\n--- OneR Top 15 Features (by AUROC on TEST) ---")
print(result_df[["feature", "roc_auc_score"]].head(15))

useful_features_df = result_df.loc[result_df["roc_auc_score"] > 0.5]
print(f"\n{len(useful_features_df)} / {len(feature_columns)} features have AUROC > 0.5.")

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

    # Threshold with Youden's J on TRAIN
    fpr, tpr, thresholds = roc_curve(y_train, ensemble_preds_train)
    J = tpr - fpr
    best_thresh_index = np.argmax(J)
    best_thresh = thresholds[best_thresh_index]

    print(f"Best threshold (found on TRAIN set): {best_thresh:.6f}")

    final_preds_binary = (ensemble_preds_test > best_thresh).astype(int)

    final_auroc = roc_auc_score(y_true=y_test, y_score=ensemble_preds_test)
    final_precision = precision_score(y_true=y_test, y_pred=final_preds_binary, zero_division=0)
    final_recall    = recall_score(y_true=y_test, y_pred=final_preds_binary, zero_division=0)
    final_f1        = f1_score(y_true=y_test, y_pred=final_preds_binary, zero_division=0)

    print("\n--- FINAL ENSEMBLE OneR METRICS (on TEST SET) ---")
    print(f"ROC-AUC:   {final_auroc:.6f}")
    print(f"Precision: {final_precision:.6f}")
    print(f"Recall:    {final_recall:.6f}")
    print(f"F1-Score:  {final_f1:.6f}")

    print("\n" + "="*60)
    print("--- 6. FINAL ANALYSIS & JUSTIFICATION ---")
    print("="*60)

    print("** CRITICAL FINDING **")
    print(f"Ensemble OneR AUROC on Distrinet-CIC-IDS2017 (this split): {final_auroc:.6f}")
else:
    print("No useful (AUROC > 0.5) features were found. Either the dataset is broken,")
    print("or your feature list does not match the CSV columns.")
