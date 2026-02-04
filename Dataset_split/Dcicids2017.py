"""
CIC-IDS2017 Parquet to CSV conversion script.

Converts Distrinet-CIC-IDS2017 Parquet files to unified multiclass and binary CSV formats.

Inputs:
  - Directory containing Distrinet-CIC-IDS2017 Parquet files (default: ../../Distrinet-CICIDS2017)

Outputs:
  - Multiclass CSV (default: ../Datasets/DCIC2017/DMcic2017.csv)
  - Binary CSV (default: ../Datasets/DCIC2017/DBcic2017.csv)

Usage:
  python Dcicids2017.py
  python Dcicids2017.py --input_dir /path/to/Distrinet-CICIDS2017 --output_dir ../Datasets/DCIC2017

Note: Requires pyarrow or fastparquet: pip install pyarrow
"""

import os
import glob
import pandas as pd
import argparse

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Convert Distrinet-CIC-IDS2017 Parquet files to CSV')
parser.add_argument('--input_dir', type=str, default=os.path.join("..", "..", "Distrinet-CICIDS2017"),
                    help='Directory containing Distrinet-CIC-IDS2017 Parquet files')
parser.add_argument('--output_dir', type=str, default='../Datasets/DCIC2017',
                    help='Output directory for CSV files')
args = parser.parse_args()

# === SETTINGS ===
folder = args.input_dir
os.makedirs(args.output_dir, exist_ok=True)

out_multiclass = os.path.join(args.output_dir, "DMcic2017.csv")
out_binary = os.path.join(args.output_dir, "DBcic2017.csv")

print(f"Input directory: {folder}")
print(f"Output directory: {args.output_dir}")

# Note: You may need to install a library to read Parquet files.
# Open your terminal or command prompt and run:
# pip install pyarrow
# or
# pip install fastparquet

# === 1. FIND AND READ ALL PARQUET FILES ===
# Update the pattern to search for .parquet files
parquet_files = glob.glob(os.path.join(folder, "*.parquet"))
if not parquet_files:
    print(f"No .parquet files found in the folder: {folder}")
    exit()

print(f"Found {len(parquet_files)} Parquet files.")

df_list = []
for f in parquet_files:
    print(f"Reading: {f}")
    # Use pd.read_parquet to read the file
    df = pd.read_parquet(f)
    df_list.append(df)

print("Concatenating DataFrames...")
df_all = pd.concat(df_list, ignore_index=True)
print(f"Total rows: {len(df_all)}")

# === 2. CLEAN COLUMNS ===
# Remove leading/trailing whitespace from column names
df_all.columns = [col.strip() for col in df_all.columns]

# Unify the label column name (e.g., "Label" vs "label")
if "Label" in df_all.columns:
    df_all.rename(columns={"Label": "label"}, inplace=True)
if "label" not in df_all.columns:
    raise ValueError("No 'label' or 'Label' column found in the combined data!")

# === 3. SAVE MULTICLASS CSV ===
print(f"Saving multiclass CSV to {out_multiclass}...")
df_all.to_csv(out_multiclass, index=False)
print("Save complete.")

# === 4. CREATE AND SAVE BINARY LABEL CSV ===
# Common benign labels are mapped to 0, all other labels (attacks) are mapped to 1.
benign_values = ['BENIGN', 'Benign', 'benign', 'Normal']
df_all['label_binary'] = df_all['label'].apply(lambda x: 0 if str(x).strip() in benign_values else 1)

# Prepare columns for the binary output file
# This creates a new DataFrame with the original label column removed and the new binary label column added.
binary_cols = [col for col in df_all.columns if col != 'label']
# The line below was reordered to place the new binary label at the end, which is more conventional.
binary_df = df_all[[col for col in binary_cols if col != 'label_binary'] + ['label_binary']]


print(f"Saving binary-label CSV to {out_binary}...")
binary_df.to_csv(out_binary, index=False)
print("Save complete.")


# === 5. PRINT LABEL DISTRIBUTION ===
print("\n--- Final Data Summary ---")
print("\nMulticlass distribution:")
print(df_all['label'].value_counts())
print("\nBinary distribution:")
print(df_all['label_binary'].value_counts())
