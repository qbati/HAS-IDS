import pandas as pd
import numpy as np

def calculate_per_class_scores(df):
    """
    Calculates the performance score for each class based on binary predictions.
    - For attack types, it calculates Recall (True Positive Rate).
    - For the 'Normal' class, it calculates Specificity (True Negative Rate).
    """
    # Get all unique classes from the true labels, including 'Normal'
    all_classes = df['true_label'].unique()
    
    print(f"\nFound the following classes to evaluate: {all_classes}")

    results = {}

    # Loop through each class to calculate its score
    for class_name in all_classes:
        
        # Isolate all samples that belong to the current class
        class_samples_df = df[df['true_label'] == class_name]
        
        if len(class_samples_df) == 0:
            continue

        total_samples_in_class = len(class_samples_df)

        if class_name == 'Normal':
            # For the 'Normal' class, a correct prediction is 0 (Normal).
            # This calculates the True Negative Rate (Specificity).
            correct_predictions = class_samples_df[class_samples_df['prediction'] == 0].shape[0]
        else:
            # For any attack class, a correct prediction is 1 (Abnormal).
            # This calculates the True Positive Rate (Recall).
            correct_predictions = class_samples_df[class_samples_df['prediction'] == 1].shape[0]
            
        # The formula is the same, but the definition of "correct" changes based on the class.
        score = correct_predictions / total_samples_in_class if total_samples_in_class > 0 else 0
        results[class_name] = score
        
    return results

# --- 1. Load Your Data ---
print("Loading data for multi-class evaluation...")

try:
    # Load the binary predictions saved by training.py
    y_pred_df = pd.read_csv('final_predictions_run.csv', header=None, names=['prediction'])

    # Load the original UNSW-NB15 test set with multi-class labels
    # Make sure the file path and column name ('attack_cat') are correct
    y_true_df = pd.read_csv('UNSW_pre_data/MUNSWTest.csv') 
except FileNotFoundError as e:
    print(f"Error: Could not find a required file. Please ensure 'final_predictions.csv' and 'UNSW_pre_data/MUNSWTest.csv' exist.")
    print(f"Details: {e}")
    exit()


# --- 2. Combine and Evaluate ---
# Ensure the number of predictions matches the number of true labels
if len(y_pred_df) != len(y_true_df):
    print(f"Error: Mismatch in number of rows. Predictions: {len(y_pred_df)}, True Labels: {len(y_true_df)}")
    exit()

evaluation_df = pd.DataFrame({
    'true_label': y_true_df['attack_cat'],
    'prediction': y_pred_df['prediction']
})

# Use the updated function to get scores for all classes
class_scores = calculate_per_class_scores(evaluation_df)

# --- 3. Display the Results ---
print("\n--- Performance Rate (%) of Each Class ---")
print("---------------------------------------------")
# Note: For attacks, this is Recall. For Normal, it's Specificity.
print(f"{'Class Type':<20} | {'Score':<10}")
print("---------------------------------------------")

# Sort the results for consistent output, placing 'Normal' first if it exists
sorted_scores = sorted(class_scores.items(), key=lambda item: (item[0] != 'Normal', item[0]))

for class_type, score in sorted_scores:
    print(f"{class_type:<20} | {score * 100:.2f}%")

print("---------------------------------------------")

