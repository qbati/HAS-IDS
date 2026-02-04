"""
Binary split script for CIC-IDS2017 dataset.

Converts multiclass train/test CSV files to binary (0=Normal, 1=Attack) format.

Inputs:
  - Multiclass training CSV (default: ../Datasets/DCIC2017/DMcic2017_train.csv)
  - Multiclass test CSV (default: ../Datasets/DCIC2017/DMcic2017_test.csv)

Outputs:
  - Binary training CSV (default: ../Datasets/DCIC2017/DBcic2017_train.csv)
  - Binary test CSV (default: ../Datasets/DCIC2017/DBcic2017_test.csv)

Usage:
  python DBsplit.py
  python DBsplit.py --train_input path/to/train.csv --test_input path/to/test.csv --output_dir path/to/output
"""

import pandas as pd
import argparse
import os

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Convert multiclass CIC-IDS2017 CSV to binary format')
parser.add_argument('--train_input', type=str, default='../Datasets/DCIC2017/DMcic2017_train.csv',
                    help='Path to multiclass training CSV')
parser.add_argument('--test_input', type=str, default='../Datasets/DCIC2017/DMcic2017_test.csv',
                    help='Path to multiclass test CSV')
parser.add_argument('--output_dir', type=str, default='../Datasets/DCIC2017',
                    help='Output directory for binary CSV files')
args = parser.parse_args()

# === SETTINGS ===
train_multiclass_file = args.train_input
test_multiclass_file = args.test_input
os.makedirs(args.output_dir, exist_ok=True)

# Define the names for the final binary output files
train_binary_file = os.path.join(args.output_dir, "DBcic2017_train.csv")
test_binary_file = os.path.join(args.output_dir, "DBcic2017_test.csv")

# === 1. PROCESS THE TRAINING FILE ===
print(f"Processing training file: {train_multiclass_file}")
try:
    # Load the precisely split training data
    train_df = pd.read_csv(train_multiclass_file)

    # Create the binary label: 0 for 'Normal', 1 for all other (malicious) labels
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    # Save the new binary-labeled training file
    train_df.to_csv(train_binary_file, index=False)
    print(f"Successfully saved binary training set to: {train_binary_file}")
    print(f"Train set shape: {train_df.shape}")
    print("Train set label distribution:")
    print(train_df['label'].value_counts())

except FileNotFoundError:
    print(f"Error: File not found -> {train_multiclass_file}. Please run the manual multiclass split script first.")

print("\n" + "="*40 + "\n")

# === 2. PROCESS THE TESTING FILE ===
print(f"Processing testing file: {test_multiclass_file}")
try:
    # Load the precisely split testing data
    test_df = pd.read_csv(test_multiclass_file)

    # Create the binary label
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'Normal' else 1)

    # Save the new binary-labeled testing file
    test_df.to_csv(test_binary_file, index=False)
    print(f"Successfully saved binary testing set to: {test_binary_file}")
    print(f"Test set shape: {test_df.shape}")
    print("Test set label distribution:")
    print(test_df['label'].value_counts())

except FileNotFoundError:
    print(f"Error: File not found -> {test_multiclass_file}. Please run the manual multiclass split script first.")

