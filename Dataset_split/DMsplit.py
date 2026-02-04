"""
Multiclass train/test split for CIC-IDS2017 dataset.

Splits the multiclass CIC-IDS2017 CSV into train/test sets matching the exact
per-class counts specified in the HAS-IDS paper (Table 2).

Inputs:
  - Multiclass CIC-IDS2017 CSV (default: ../Datasets/DCIC2017/DMcic2017.csv)

Outputs:
  - Training CSV (default: ../Datasets/DCIC2017/DMcic2017_train.csv)
  - Test CSV (default: ../Datasets/DCIC2017/DMcic2017_test.csv)

Usage:
  python DMsplit.py
  python DMsplit.py --input path/to/DMcic2017.csv --output_dir path/to/output
"""

import pandas as pd
import numpy as np
import argparse
import os

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Split CIC-IDS2017 multiclass CSV into train/test sets')
parser.add_argument('--input', type=str, default='../Datasets/DCIC2017/DMcic2017.csv',
                    help='Path to multiclass CIC-IDS2017 CSV file')
parser.add_argument('--output_dir', type=str, default='../Datasets/DCIC2017',
                    help='Output directory for train/test CSV files')
args = parser.parse_args()

# === SETTINGS ===
csv_file = args.input
os.makedirs(args.output_dir, exist_ok=True)

# === 1. LOAD THE FULL, CORRECTED DATASET ===
try:
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {csv_file}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found. Please run the data correction script first.")
    exit()

# === 2. DEFINE THE EXACT SPLIT COUNTS FROM THE PAPER (TABLE 2) ===
# These are the precise numbers of samples required for the train set for each category
train_counts = {
    'Normal': 923528,
    'Botnet': 580,
    'DDoS': 76173,
    'DoS GoldenEye': 6170,
    'DoS Hulk': 126876,
    'DoS Slowhttptest': 1394,
    'DoS Slowloris': 3211,
    'FTP-Patator': 3188,
    'Heartbleed': 7,
    'Infiltration': 32,
    'Infiltration - Portscan': 4374, # Note the space difference in label name
    'Portscan': 1350,
    'SSH-Patator': 2344,
    'Web Attack - Brute Force': 60,
    'Web Attack - SQL Injection': 10,
    'Web Attack - XSS': 15
}

# Unify label names to match the keys in our dictionary
# (e.g., 'DoS-goldenEye' in data vs. 'DoS GoldenEye' in paper table)
df['label'] = df['label'].replace({
    'DoS-goldenEye': 'DoS GoldenEye',
    'DoS-hulk': 'DoS Hulk',
    'DoS-slowhttptest': 'DoS Slowhttptest',
    'DoS-slowloris': 'DoS Slowloris',
    'FTP-patator': 'FTP-Patator',
    'Infiltration-portscan': 'Infiltration - Portscan',
    'SSH-patator': 'SSH-Patator',
    'Brute force': 'Web Attack - Brute Force',
    'SQL injection': 'Web Attack - SQL Injection',
    'XSS': 'Web Attack - XSS'
})


# === 3. MANUALLY SPLIT THE DATASET ===
print("\nManually splitting dataset to match paper's distribution...")
train_dfs = []
test_dfs = []

# Set a random seed for reproducible sampling
np.random.seed(42)

for label, train_count in train_counts.items():
    # Get all data for the current label
    category_df = df[df['label'] == label]
    
    # Check if we have enough data
    if len(category_df) < train_count:
        print(f"Warning: Not enough data for '{label}'. Have {len(category_df)}, need {train_count}. Using all available for training.")
        train_sample = category_df
        test_sample = pd.DataFrame(columns=df.columns) # Empty dataframe for test
    else:
        # Randomly sample the exact number of rows for the training set
        train_sample = category_df.sample(n=train_count, random_state=42)
        # The remaining rows go to the test set
        test_sample = category_df.drop(train_sample.index)

    train_dfs.append(train_sample)
    test_dfs.append(test_sample)

# Concatenate all the small dataframes into the final train and test sets
train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"\nFinal Train shape: {train_df.shape}, Final Test shape: {test_df.shape}")

# === 4. SAVE THE PRECISELY SPLIT OUTPUT ===
train_file = os.path.join(args.output_dir, "DMcic2017_train.csv")
test_file = os.path.join(args.output_dir, "DMcic2017_test.csv")

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"\nSaved train set: {train_file}")
print(f"Saved test set : {test_file}")

# === 5. VERIFY THE FINAL LABEL BALANCE ===
print("\n--- Final Train Set Distribution (Verification) ---")
print(train_df['label'].value_counts())
print("\n--- Final Test Set Distribution (Verification) ---")
print(test_df['label'].value_counts())
