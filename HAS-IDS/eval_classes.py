import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, precision_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# COMMAND-LINE INTERFACE CONFIGURATION
# ==============================================================================
# This script evaluates per-class recall metrics for HAS-IDS predictions.
# Required inputs:
#   --binary: Path to binary predictions CSV (columns: binary_label, binary_pred)
#   --multi:  Path to multiclass labels CSV (column: attack_cat)
# ==============================================================================

import argparse

parser = argparse.ArgumentParser(
    description='Evaluate per-class recall for HAS-IDS predictions',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Example usage:
  python eval_classes.py --binary unsw_nb15_hasids_test_predictions.csv --multi ../Datasets/UNSW/MUNSWTest.csv
    '''
)
parser.add_argument('--binary', type=str, required=True, 
                    help='Path to binary predictions CSV (columns: binary_label, binary_pred)')
parser.add_argument('--multi', type=str, required=True,
                    help='Path to multiclass labels CSV (column: attack_cat)')
args = parser.parse_args()

binary_csv_path = args.binary
multiclass_csv_path = args.multi

print("Initiating per-class recall evaluation...")
print(f"Binary predictions source: {binary_csv_path}")
print(f"Multiclass labels source: {multiclass_csv_path}")

# ========== LOAD BINARY PREDICTION RESULTS ==========
try:
    binary_df = pd.read_csv(binary_csv_path)
    if not {'binary_label', 'binary_pred'}.issubset(binary_df.columns):
        raise ValueError("Missing 'binary_label' or 'binary_pred' in binary CSV.")
except Exception as e:
    print(f"❌ Error loading binary predictions: {e}")
    exit()

# ========== LOAD MULTICLASS LABELS ==========
try:
    multi_df = pd.read_csv(multiclass_csv_path)
    if 'attack_cat' not in multi_df.columns:
        raise ValueError("Missing 'attack_cat' column in multiclass CSV.")
except Exception as e:
    print(f"❌ Error loading multiclass labels: {e}")
    exit()

# ========== MERGE DATA ==========
if len(binary_df) != len(multi_df):
    print(f"❌ Row count mismatch: binary={len(binary_df)}, multiclass={len(multi_df)}")
    exit()

results_df = pd.concat([binary_df, multi_df['attack_cat']], axis=1)
results_df['attack_cat'] = results_df['attack_cat'].astype(str).str.strip()

# ========== OVERALL PERFORMANCE METRICS ==========
print("\n==============================")
print("OVERALL PERFORMANCE METRICS")
print("==============================")

y_true = results_df['binary_label']
y_pred = results_df['binary_pred']


acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# ========== PER-CLASS RECALL ANALYSIS ==========
print("\n==============================")
print("PER-CLASS RECALL ANALYSIS")
print("==============================")

recall_per_category = {}
all_categories = sorted(results_df['attack_cat'].unique())

for category in all_categories:
    category_df = results_df[results_df['attack_cat'] == category]
    true_slice = category_df['binary_label']
    pred_slice = category_df['binary_pred']
    

    if category.lower() == 'normal':
        # Specificity for benign class
        tn = ((true_slice == 0) & (pred_slice == 0)).sum()
        total = len(category_df)
        specificity = tn / total if total > 0 else 0
        recall_per_category[category] = specificity
    else:
        # Recall for attack classes
        rec = recall_score(true_slice, pred_slice, zero_division=0)
        recall_per_category[category] = rec

# Print the per-class recall/specificity
print("Recall Rate (%) for each class:\n")
sorted_categories = sorted(recall_per_category.keys(), key=lambda x: (x.lower() != 'normal', x.lower()))

for category in sorted_categories:
    print(f"{category:<20}: {recall_per_category[category] * 100:.2f}%")
