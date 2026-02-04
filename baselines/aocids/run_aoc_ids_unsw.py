import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score
from utils_aoc import load_data, SplitData, AE, CRCLoss, setup_seed, evaluate
import time
import os
import json
import argparse
import warnings

warnings.filterwarnings('ignore')

# -------------------------
# 1. CONFIGURATION
# -------------------------
# These are the *paper's* HPs for UNSW-NB15
dataset = 'unsw'
epochs = 300           # epoch_0
epoch_1 = 3            # epoch_1
percent = 0.8          # This is test_size, so 1.0 - 0.8 = 0.2 initial train
flip_percent = 0.05    # λ = 5%
sample_interval = 2784
cuda_num = "0"
input_dim = 196

# These are from the original script
tem = 0.02
bs = 128
seed = 5009
seed_round = 5

# --- Our Bake-off Paths ---
UNSWTrain_dataset_path = "../Datasets/UNSW/BUNSWTrain.csv"
UNSWTest_dataset_path = "../Datasets/UNSW/BUNSWTest.csv"
ORIGINAL_TEST_PATH_FOR_RECALL = "../Datasets/UNSW/MUNSWTest.csv"
RESULTS_DIR = "results_aoc_github"
RUN_NAME = "aoc_ids_github_unsw"
# -------------------------

# 2. SETUP
# -------------------------
device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs(RESULTS_DIR, exist_ok=True)
criterion = CRCLoss(device, tem)
all_run_metrics = [] # To store metrics from all 5 runs

# 3. DATA LOADING
# -------------------------
print("--- Loading and Preprocessing Data (AOC-IDS GitHub Method) ---")
UNSWTrain = load_data(UNSWTrain_dataset_path)
UNSWTest = load_data(UNSWTest_dataset_path)

# Load recall file to check if it exists
try:
    df_recall_data_check = pd.read_csv(ORIGINAL_TEST_PATH_FOR_RECALL)
    print("Recall file found.")
except FileNotFoundError:
    print(f"Warning: Recall file not found at {ORIGINAL_TEST_PATH_FOR_RECALL}")
    print("Per-class recall will be skipped.")
    df_recall_data_check = None

# Create an instance of SplitData for 'unsw'
splitter_unsw = SplitData(dataset='unsw')

# Transform the data (Note: This is the paper's original method)
x_train, y_train = splitter_unsw.transform(UNSWTrain, labels='label')
x_test, y_test = splitter_unsw.transform(UNSWTest, labels='label')

# Convert to torch tensors
x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# -------------------------
# 4. START 5-RUN LOOP
# -------------------------
for i in range(seed_round):
    current_seed = seed + i
    print(f"\n{'='*30} STARTING RUN {i+1}/{seed_round} (Seed: {current_seed}) {'='*30}")
    setup_seed(current_seed)
    
    run_metrics = {} # Store metrics for this specific run

    # --- Initial Data Split ---
    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(
        x_train, y_train, test_size=percent, random_state=current_seed, stratify=y_train
    )
    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=bs, shuffle=True)
    
    num_of_first_train = online_x_train.shape[0]

    model = AE(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

    # --- Start Training Timer ---
    train_start_time = time.perf_counter()
    
    model.train()
    for epoch in range(epochs):
        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f'  Seed = {current_seed}, first round: epoch = {epoch+1}/{epochs}')
            
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            features, recon_vec = model(inputs)
            loss = criterion(features,labels) + criterion(recon_vec,labels)
            
            if not torch.isnan(loss) and loss > 0:
                loss.backward()
                optimizer.step()

    # Prep for online loop
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    online_x_train, online_y_train  = online_x_train.to(device), online_y_train.to(device)
    x_train_this_epoch, x_test_left_epoch = online_x_train.clone(), online_x_test.clone().to(device)
    y_train_this_epoch = online_y_train.clone()

    # --- Start Online Training ---
    count = 0
    y_train_detection = y_train_this_epoch.clone()
    total_sim_chunks = int(np.ceil(len(x_test_left_epoch) / sample_interval))
    
    while len(x_test_left_epoch) > 0:
        count += 1
        
        if len(x_test_left_epoch) < sample_interval:
            x_test_this_epoch = x_test_left_epoch.clone()
            x_test_left_epoch.resize_(0)
        else:
            x_test_this_epoch = x_test_left_epoch[:sample_interval].clone()
            x_test_left_epoch = x_test_left_epoch[sample_interval:]

        model.eval() # Switch to eval mode for pseudo-labeling
        
        # must compute the normal_temp... again, because the model has been updated
        with torch.no_grad():
            initial_normal_data = online_x_train[(online_y_train == 0).squeeze()]
            normal_temp = torch.mean(F.normalize(model(initial_normal_data)[0], p=2, dim=1), dim=0)
            normal_recon_temp = torch.mean(F.normalize(model(initial_normal_data)[1], p=2, dim=1), dim=0)
        
        # `evaluate` returns pseudo-labels (y_pred_final) when y_test=0
        predict_label = evaluate(normal_temp, normal_recon_temp, x_train_this_epoch, y_train_detection, x_test_this_epoch, 0, model)

        y_test_pred_this_epoch = predict_label
        y_train_detection = torch.cat((y_train_detection.to(device), torch.tensor(y_test_pred_this_epoch).to(device)))
        
        # --- Flip pseudo-labels ---
        y_test_pred_flipped = y_test_pred_this_epoch.copy()
        num_zero = int(flip_percent * y_test_pred_this_epoch.shape[0])
        if num_zero > 0:
            zero_indices = np.random.choice(y_test_pred_this_epoch.shape[0], num_zero, replace=False)
            y_test_pred_flipped[zero_indices] = 1 - y_test_pred_flipped[zero_indices]

        x_train_this_epoch = torch.cat((x_train_this_epoch.to(device), x_test_this_epoch.to(device)))
        y_train_this_epoch_temp = y_train_this_epoch.clone()
        y_train_this_epoch = torch.cat((y_train_this_epoch_temp.to(device), torch.tensor(y_test_pred_flipped).to(device)))

        train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds, batch_size=bs, shuffle=True)
        
        model.train() # Switch back to train mode
        for epoch in range(epoch_1):
            if count % 10 == 0:
                print(f'  Seed = {current_seed}, Online Round {count}/{total_sim_chunks}, Finetune Epoch {epoch+1}/{epoch_1}')
            
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                features, recon_vec = model(inputs)
                loss = criterion(features,labels) + criterion(recon_vec,labels)
                
                if not torch.isnan(loss) and loss > 0:
                    loss.backward()
                    optimizer.step()

    # --- Stop Training Timer ---
    train_end_time = time.perf_counter()
    T_train = train_end_time - train_start_time
    print(f"--- [Run {i+1}] Training Finished. T_train = {T_train:.4f}s ---")

    # -------------------------
    # 5. FINAL EVALUATION (BAKE-OFF)
    # -------------------------
    print(f"--- [Run {i+1}] Starting Final Evaluation ---")
    test_start_time = time.perf_counter()
    
    model.eval()
    with torch.no_grad():
        # Get final templates from *initial* normal data
        initial_normal_data = online_x_train[(online_y_train == 0).squeeze()]
        normal_temp = torch.mean(F.normalize(model(initial_normal_data)[0], p=2, dim=1), dim=0)
        normal_recon_temp = torch.mean(F.normalize(model(initial_normal_data)[1], p=2, dim=1), dim=0)

    # `evaluate` returns full results when y_test is provided
    # y_train_detection contains all *pseudo-labels* used for fitting gaussians
    y_pred_final, y_score_final, (acc, prec, rec, f1, cm) = evaluate(
        normal_temp, normal_recon_temp, 
        x_train_this_epoch, y_train_detection, 
        x_test, y_test, model
    )
    
    test_end_time = time.perf_counter()
    T_test = test_end_time - test_start_time
    print(f"--- [Run {i+1}] Evaluation Finished. T_test = {T_test:.4f}s ---")

    # --- 6. METRICS CALCULATION (BAKE-OFF) ---
    print(f"--- [Run {i+1}] Calculating Final Metrics ---")
    y_test_np = y_test.cpu().numpy()
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    roc = roc_auc_score(y_test_np, y_score_final)
    pr = average_precision_score(y_test_np, y_score_final)

    print(f"[Run {i+1}] Accuracy:  {acc:.4f}")
    print(f"[Run {i+1}] Precision: {prec:.4f}")
    print(f"[Run {i+1}] Recall:    {rec:.4f}")
    print(f"[Run {i+1}] F1 Score:  {f1:.4f}")
    print(f"[Run {i+1}] FPR:       {fpr:.4f}")
    print(f"[Run {i+1}] ROC-AUC:   {roc:.4f}")
    print(f"[Run {i+1}] PR-AUC:    {pr:.4f}")
    print(f"[Run {i+1}] Confusion Matrix:\n{cm}")

    # --- Per-Class Recall ---
    print(f"\n--- [Run {i+1}] Per-Class Recall (from original test CSV) ---")
    per_class_recall_results = {}
    try:
        df_recall_data = pd.read_csv(ORIGINAL_TEST_PATH_FOR_RECALL)
        
        if 'attack_cat' not in df_recall_data.columns:
            print(f"Error: 'attack_cat' column not found in {ORIGINAL_TEST_PATH_FOR_RECALL}.")
        elif len(df_recall_data) != len(y_test_np):
            print(f"Error: Row count mismatch! Test set has {len(y_test_np)} rows, recall file has {len(df_recall_data)}.")
        else:
            df_recall_data['y_pred'] = y_pred_final
            y_true_recall = df_recall_data['label'].values.astype(int)
            all_categories = df_recall_data['attack_cat'].unique()
            
            for cat_name in sorted(all_categories):
                cat_mask = (df_recall_data['attack_cat'] == cat_name)
                y_true_subset = y_true_recall[cat_mask]
                y_pred_subset = y_pred_final[cat_mask]
                support = len(y_true_subset)
                if support == 0: continue
                if cat_name == 'Normal':
                    cat_recall = recall_score(y_true_subset, y_pred_subset, pos_label=0, zero_division=0)
                else:
                    cat_recall = recall_score(y_true_subset, y_pred_subset, pos_label=1, zero_division=0)
                print(f"  - {cat_name:<18}: Recall = {cat_recall:.4f} (Support: {support})")
                per_class_recall_results[cat_name] = {'recall': cat_recall, 'support': support}
    except Exception as e:
        print(f"Skipping per-class recall due to error: {e}")

    # --- 7. SAVE RESULTS (BAKE-OFF) ---
    print(f"\n--- [Run {i+1}] Saving Results ---")
    run_save_name = f"{RUN_NAME}_seed{current_seed}"
    
    pred_df = pd.DataFrame({"y_true": y_test_np, "y_pred": y_pred_final, "y_score": y_score_final})
    pred_path = os.path.join(RESULTS_DIR, f"{run_save_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    metrics = {
        "model": "AOC-IDS-GitHub", "dataset": "UNSW-NB15", "seed": current_seed,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
        "roc_auc": roc, "pr_auc": pr, "cm": cm.tolist(),
        "t_train_sec": T_train, "t_test_sec": T_test,
        "per_class_recall": per_class_recall_results
    }
    all_run_metrics.append(metrics)
    
    metrics_path = os.path.join(RESULTS_DIR, f"{run_save_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"--- RUN {i+1} COMPLETE ---")

# -------------------------
# 8. FINAL AVERAGED RESULTS
# -------------------------
print(f"\n{'='*30} ALL 5 RUNS COMPLETE - AVERAGED RESULTS {'='*30}")
avg_metrics = {}
for key in all_run_metrics[0].keys():
    if isinstance(all_run_metrics[0][key], (int, float)):
        values = [m[key] for m in all_run_metrics]
        avg_metrics[f"avg_{key}"] = np.mean(values)
        avg_metrics[f"std_{key}"] = np.std(values)

print(json.dumps(avg_metrics, indent=2))
avg_metrics_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}_metrics_AVERAGED.json")
with open(avg_metrics_path, 'w') as f:
    json.dump(avg_metrics, f, indent=2)
print(f"\nAveraged metrics saved to {avg_metrics_path}")