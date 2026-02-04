"""
Generic stratified train/test split for CSV datasets.

Performs an 80/20 stratified train/test split on any CSV file with a label column.

Inputs:
  - CSV file with label column (default: ../Datasets/DCIC2017/DMcic2017.csv)

Outputs:
  - Training CSV (default: <input>_train.csv)
  - Test CSV (default: <input>_test.csv)

Usage:
  python split.py
  python split.py --input path/to/data.csv --test_size 0.2 --random_state 42
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Perform stratified train/test split on CSV dataset')
parser.add_argument('--input', type=str, default='../Datasets/DCIC2017/DMcic2017.csv',
                    help='Path to input CSV file')
parser.add_argument('--test_size', type=float, default=0.2,
                    help='Fraction of data for test set (default: 0.2)')
parser.add_argument('--random_state', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# === SETTINGS ===
csv_file = args.input
test_size = args.test_size
random_state = args.random_state

# === 1. LOAD DATA ===
df = pd.read_csv(csv_file)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# === 2. SET LABEL COLUMN ===
# For multiclass, the label is 'label'
# For binary, the label is 'label_binary'
if 'label_binary' in df.columns:
    label_col = 'label_binary'
else:
    label_col = 'label'

# === 3. TRAIN-TEST SPLIT (Stratified) ===
train_df, test_df = train_test_split(
    df, 
    test_size=test_size, 
    stratify=df[label_col], 
    random_state=random_state
)

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# === 4. SAVE OUTPUT ===
base = csv_file.replace('.csv', '')
train_file = f"{base}_train.csv"
test_file = f"{base}_test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Saved train set: {train_file}")
print(f"Saved test set : {test_file}")

# === 5. CHECK LABEL BALANCE ===
print("\nTrain set label distribution:")
print(train_df[label_col].value_counts())
print("\nTest set label distribution:")
print(test_df[label_col].value_counts())
