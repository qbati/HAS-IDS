import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# --- Configuration ---
RESULTS_DIR = "results_aoc_github"
RECALL_FILE_PATH = "../Datasets/UNSW/MUNSWTest.csv" 
FINAL_CSV_NAME = "aoc_ids_github_unsw_predictions_AVERAGED.csv"

# --- End Configuration ---

def main():
    print("--- Starting Averaging Script for AOC-IDS ---")
    
    # 1. Find all prediction files
    search_path = os.path.join(RESULTS_DIR, "aoc_ids_github_unsw_seed*_predictions.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"Error: No prediction files found at '{search_path}'")
        return
        
    print(f"Found {len(csv_files)} prediction files to average:")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    # 2. Load all y_score columns
    all_scores = []
    binary_label = None
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            all_scores.append(df['y_score'])
            if binary_label is None:
                # Use the binary_label from the first file as our base
                binary_label = df['y_true']
        except Exception as e:
            print(f"Error reading {f}: {e}")
            return

    # 3. Create the averaged DataFrame
    avg_df = pd.DataFrame()
    avg_df['binary_label'] = binary_label
    avg_df['y_score_avg'] = np.mean(all_scores, axis=0)
    
    # 4. Define threshold (0.5) and make new predictions
    THRESHOLD = 0.5 
    avg_df['binary_pred'] = (avg_df['y_score_avg'] >= THRESHOLD).astype(int)

    # 5. Load recall data and merge
    try:
        df_recall = pd.read_csv(RECALL_FILE_PATH)
        
        if 'attack_cat' not in df_recall.columns:
            raise Exception("Recall file missing 'attack_cat' column")
        
        if 'label' in df_recall.columns:
            if not np.all(avg_df['binary_label'] == df_recall['label']):
                print("--- !!! WARNING !!! ---")
                print("Warning: 'binary_label' in prediction CSVs does not match 'label' in recall file.")
                avg_df['binary_label'] = df_recall['label']
        else:
            print("--- !!! WARNING !!! ---")
            print("Recall file does not contain a 'label' column. Using 'binary_label' from prediction files.")
        
        avg_df['attack_cat'] = df_recall['attack_cat']
        
    except Exception as e:
        print(f"\n--- !!! WARNING !!! ---")
        print(f"Error loading recall file: {e}")
        avg_df['attack_cat'] = 'Unknown'

    # 6. Save the final averaged CSV
    final_path = os.path.join(RESULTS_DIR, FINAL_CSV_NAME)
    avg_df.to_csv(final_path, index=False)
    print(f"\nSuccessfully created averaged prediction file at: {final_path}")

    # 7. Calculate and print all final metrics
    binary_label_np = avg_df['binary_label']
    binary_pred_np = avg_df['binary_pred']
    y_score_np = avg_df['y_score_avg']

    acc = accuracy_score(binary_label_np, binary_pred_np)
    prec = precision_score(binary_label_np, binary_pred_np, zero_division=0)
    rec = recall_score(binary_label_np, binary_pred_np, zero_division=0)
    f1 = f1_score(binary_label_np, binary_pred_np, zero_division=0)
    cm = confusion_matrix(binary_label_np, binary_pred_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    try:
        roc = roc_auc_score(binary_label_np, y_score_np)
        pr = average_precision_score(binary_label_np, y_score_np)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC scores. {e}")
        roc = 0.0
        pr = 0.0

    print("\n--- Final Metrics Based on Averaged Predictions ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 8. Calculate and print final per-class recall, including Normal class
    print("\n--- Final Per-Class Recall (from Averaged Predictions) ---")
    print(f"{'Category':<18} | {'Recall':<10} | {'Support':<10}")
    print("-" * 42)
    
    # ** Normal class = 0, Attack class = 1 **  
    categories_to_include = avg_df['attack_cat'].unique()
    
    for cat_name in categories_to_include:
        cat_mask = (avg_df['attack_cat'] == cat_name)
        
        y_true_subset = avg_df[cat_mask]['binary_label']
        y_pred_subset = avg_df[cat_mask]['binary_pred']
        
        support = len(y_true_subset)
        if support == 0: continue

        # Calculate recall for each category
        cat_recall = recall_score(y_true_subset, y_pred_subset, pos_label=1, zero_division=0) if cat_name != 'Normal' else recall_score(y_true_subset, y_pred_subset, pos_label=0, zero_division=0)
        
        print(f"{cat_name:<18} | {cat_recall:<10.4f} | {support:<10}")

if __name__ == "__main__":
    main()
