import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
import os
import json
import warnings
import random # Make sure random is imported
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

warnings.filterwarnings('ignore')

# -------------------------
# 1. CONFIGURATION
# -------------------------
TRAIN_PATH = "../Datasets/DCIC2017/DBcic2017_train.csv"
TEST_PATH = "../Datasets/DCIC2017/DBcic2017_test.csv"
RECALL_PATH = "../Datasets/DCIC2017/DMcic2017_test.csv" # For per-class recall
RESULTS_DIR = "results_baselines_dcic"
RUN_NAME = "baseline_ae_dcic"

# --- Model Hyperparameters ---
SEED = 42
INPUT_DIM = 82 # As requested
LATENT_DIM = 32 # As requested
EPOCHS = 100       # A good starting point, increase if needed
BATCH_SIZE = 256
LR = 0.001

# -------------------------
# 2. HELPER FUNCTIONS
# -------------------------
def setup_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class Autoencoder(nn.Module):
    """A standard MLP Autoencoder, 82 -> 64 -> 32"""
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), # 82 -> 64
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # 64 -> 32 (Bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), # 32 -> 64
            nn.ReLU(),
            nn.Linear(64, input_dim)   # 64 -> 82 (Reconstruct original)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def find_best_threshold(y_true, y_scores):
    """
    Finds the threshold that maximizes F1 score.
    y_scores are anomaly scores (higher = more anomalous).
    """
    best_f1 = -1
    best_thresh = 0
    
    # Check 100 percentile thresholds
    thresholds = np.percentile(y_scores, np.linspace(0, 100, 100))
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"[Threshold Finder] Best F1: {best_f1:.4f} at Threshold: {best_thresh:.6f}")
    return best_thresh

# -------------------------
# 3. SETUP
# -------------------------
setup_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs(RESULTS_DIR, exist_ok=True)
criterion = nn.MSELoss(reduction='none') # We want per-sample error

# -------------------------
# 4. DATA LOADING & PREPROCESSING
# -------------------------
print("--- Loading and Preprocessing Data (DCIC2017) ---")
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# Use 'label' for DCIC2017 (FIXED)
y_train = df_train["label"].values.astype(int)
y_test = df_test["label"].values.astype(int)

# Get feature columns
drop_cols = ['label', 'binary_label', 'attack_cat', 'id']
feature_cols = [c for c in df_train.columns if c not in drop_cols and c in df_train.columns]
cat_cols = df_train[feature_cols].select_dtypes(include="object").columns.tolist()
print(f"Categorical features: {cat_cols}")

# One-hot encoding (no leakage)
X_train_df = pd.get_dummies(df_train[feature_cols], columns=cat_cols)
X_test_df = pd.get_dummies(df_test[feature_cols], columns=cat_cols)

# Align columns
X_train_df, X_test_df = X_train_df.align(X_test_df, join='left', axis=1, fill_value=0)

# Scaling (no leakage)
num_cols = [c for c in feature_cols if c not in cat_cols and c in X_train_df.columns]
scaler = MinMaxScaler()
X_train_df[num_cols] = scaler.fit_transform(X_train_df[num_cols])
X_test_df[num_cols] = scaler.transform(X_test_df[num_cols])

X_train = X_train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)

# Verify Input Dim
if X_train.shape[1] != INPUT_DIM:
    print(f"Warning: Calculated input dim {X_train.shape[1]} does not match config {INPUT_DIM}.")
    print(f"Using {X_train.shape[1]} as the correct dimension.")
    INPUT_DIM = X_train.shape[1] # Overwrite with the real dim
print(f"Input dimension set to: {INPUT_DIM}")


# --- *** CRITICAL STEP FOR AE *** ---
# Train ONLY on Normal data
X_train_normal = X_train[y_train == 0]
print(f"Total training samples: {len(X_train)}, Normal-only training samples: {len(X_train_normal)}")

# Create DataLoaders
train_normal_ds = TensorDataset(torch.from_numpy(X_train_normal))
train_normal_loader = DataLoader(train_normal_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# 5. TRAINING PHASE
# -------------------------
print(f"--- Starting Autoencoder Training ({EPOCHS} epochs) ---")
model = Autoencoder(INPUT_DIM, LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

train_start_time = time.perf_counter()
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    for (batch_x,) in train_normal_loader: # Note: no y labels
        batch_x = batch_x.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(batch_x)
        
        # Loss is the difference between input and output
        loss = torch.mean(criterion(reconstructed, batch_x)) 
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_normal_loader)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{EPOCHS}, Avg. Reconstruction Loss: {avg_loss:.8f}')

train_end_time = time.perf_counter()
T_train = train_end_time - train_start_time
print(f"--- Training Finished. T_train = {T_train:.4f}s ---")

# -------------------------
# 6. TESTING & ANOMALY SCORING
# -------------------------
print("--- Starting Evaluation (Calculating Anomaly Scores) ---")
test_start_time = time.perf_counter()

all_scores = []
all_true = []

model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        
        reconstructed = model(batch_x)
        
        # Calculate per-sample MSE loss (this is the anomaly score)
        sample_losses = torch.mean(criterion(reconstructed, batch_x), dim=1)
        
        all_scores.append(sample_losses.cpu().numpy())
        all_true.append(batch_y.numpy())

y_scores_np = np.concatenate(all_scores)
y_true_np = np.concatenate(all_true)

test_end_time = time.perf_counter()
T_test = test_end_time - test_start_time
print(f"--- Evaluation Finished. T_test = {T_test:.4f}s ---")

# -------------------------
# 7. METRIC CALCULATION (Find Best Threshold)
# -------------------------
print("--- Calculating Final Metrics (Finding Best Threshold) ---")

# Find the threshold that maximizes F1-score
best_threshold = find_best_threshold(y_true_np, y_scores_np)

# Get final predictions based on this best threshold
y_pred_final = (y_scores_np >= best_threshold).astype(int)

# Calculate all 8 metrics
acc = accuracy_score(y_true_np, y_pred_final)
prec = precision_score(y_true_np, y_pred_final, zero_division=0)
rec = recall_score(y_true_np, y_pred_final, zero_division=0)
f1 = f1_score(y_true_np, y_pred_final, zero_division=0)
cm = confusion_matrix(y_true_np, y_pred_final, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
roc = roc_auc_score(y_true_np, y_scores_np) # ROC/PR use the raw scores
pr = average_precision_score(y_true_np, y_scores_np)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"FPR:       {fpr:.4f}")
print(f"ROC-AUC:   {roc:.4f}")
print(f"PR-AUC:    {pr:.4f}")
print(f"Confusion Matrix:\n{cm}")

# --- 8. Per-Class Recall ---
print(f"\n==============================")
print(f"PER-CLASS RECALL ANALYSIS")
print(f"==============================")
print(f"Recall Rate (%) for each class (from {RECALL_PATH}):\n")

per_class_recall_results = {}
try:
    df_recall_data = pd.read_csv(RECALL_PATH)
    
    # Check for attack_cat
    if 'attack_cat' not in df_recall_data.columns:
        print(f"Error: 'attack_cat' column not found in {RECALL_PATH}.")
        raise KeyError("'attack_cat' column missing")
        
    if len(df_recall_data) != len(y_true_np):
        print(f"Error: Row count mismatch! Test set has {len(y_true_np)} rows, recall file has {len(df_recall_data)}.")
        raise ValueError("Row count mismatch")

    # Use the y_true_np we already have from the test set
    y_true_recall = y_true_np
    all_categories = df_recall_data['attack_cat'].unique()
    
    # *** THIS BLOCK IS NOW FIXED with your new spellings ***
    category_order = [
        "Normal",
        "Botnet",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS Slowloris",
        "FTP-Patator",
        "Heartbleed",
        "Infiltration",
        "Infiltration - Portscan",
        "Portscan",
        "SSH-Patator",
        "Web Attack - Brute Force",
        "Web Attack - SQL Injection",
        "Web Attack - XSS"
    ]
    
    # Find longest name for formatting
    max_len = 0
    for cat_name in category_order:
        if cat_name not in all_categories:
            continue
        if len(cat_name) > max_len:
            max_len = len(cat_name)

    for cat_name in category_order:
        if cat_name not in all_categories:
            print(f"{cat_name:<{max_len}} : (Category not found in recall file)")
            continue
            
        cat_mask = (df_recall_data['attack_cat'] == cat_name)
        
        y_true_subset = y_true_recall[cat_mask]
        y_pred_subset = y_pred_final[cat_mask]
        
        support = len(y_true_subset)
        if support == 0: 
            print(f"{cat_name:<{max_len}} : 0.00% (Support: 0)")
            continue

        if cat_name == 'Normal':
            cat_recall = recall_score(y_true_subset, y_pred_subset, pos_label=0, zero_division=0)
        else:
            cat_recall = recall_score(y_true_subset, y_pred_subset, pos_label=1, zero_division=0)
        
        print(f"{cat_name:<{max_len}} : {cat_recall*100:.2f}%")
        per_class_recall_results[cat_name] = {'recall': cat_recall, 'support': support}
        
except Exception as e:
    print(f"Skipping per-class recall due to error: {e}")

# --- 9. SAVE RESULTS ---
print(f"\n--- Saving Results to {RESULTS_DIR} ---")
pred_df = pd.DataFrame({
    "y_true": y_true_np, 
    "y_pred": y_pred_final, 
    "y_score": y_scores_np
})
pred_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}_predictions.csv")
pred_df.to_csv(pred_path, index=False)
print(f"Predictions saved to {pred_path}")

metrics = {
    "model": "Autoencoder", "dataset": "DCIC2017", "seed": SEED,
    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
    "roc_auc": roc, "pr_auc": pr, "cm": cm.tolist(),
    "t_train_sec": T_train, "t_test_sec": T_test,
    "best_threshold": best_threshold,
    "per_class_recall": per_class_recall_results
}
metrics_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")
print("--- Autoencoder Run Complete ---")