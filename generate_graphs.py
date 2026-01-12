import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from setfit import SetFitModel

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATH = "xpi_labeled_data_augmented.csv"
# Point this to the folder where you successfully saved the model
MODEL_DIR = "setfit_sentiment_model_final" 
OUTPUT_DIR = "model_visualizations"

def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """
    Recreates the specific style of confusion matrix from your original script.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('SetFit Sentiment Classification Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def main():
    print("--- Generating Post-Training Visualizations ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. LOAD DATA & RECREATE SPLIT
    # We must use the EXACT same random_state=42 to ensure we test on the 
    # data the model has never seen, just like the original training plan.
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    text_col = next((col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()), 'text')
    label_col = next((col for col in df.columns if 'label' in col.lower() or 'sentiment' in col.lower()), 'label')
    df = df.dropna(subset=[text_col, label_col])
    
    print("Recreating Train/Test split (random_state=42)...")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    unique_labels = sorted(df[label_col].unique().tolist())
    print(f"Test set size: {len(test_df)} examples")

    # 2. LOAD MODEL
    print(f"Loading model from {MODEL_DIR}...")
    try:
        model = SetFitModel.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure MODEL_DIR points to the folder created by the 'finish_head_training.py' script.")
        return

    # 3. PREDICT
    print("Running predictions on test set...")
    y_true = test_df[label_col].tolist()
    y_pred = model.predict(test_df[text_col].tolist())

    # 4. GENERATE VISUALIZATIONS
    print("Plotting Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, unique_labels, OUTPUT_DIR)
    
    # 5. GENERATE METRICS REPORT
    print("Generating Classification Report...")
    report = classification_report(y_true, y_pred, target_names=[str(l) for l in unique_labels])
    print("\n" + report)
    
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print(f"Classification report saved to {OUTPUT_DIR}/classification_report.txt")

if __name__ == "__main__":
    main()