"""Generate post-training visualizations and metrics."""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from setfit import SetFitModel

DATA_PATH = "xpi_labeled_data_augmented.csv"
MODEL_DIR = "setfit_sentiment_model_final"
OUTPUT_DIR = "model_visualizations"


def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Generate and save confusion matrix visualization."""
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

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    text_col = next((col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()), 'text')
    label_col = next((col for col in df.columns if 'label' in col.lower() or 'sentiment' in col.lower()), 'label')
    df = df.dropna(subset=[text_col, label_col])
    
    # Use same random_state as training to ensure consistent test split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    unique_labels = sorted(df[label_col].unique().tolist())
    print(f"Test set size: {len(test_df)} examples")

    print(f"Loading model from {MODEL_DIR}...")
    try:
        model = SetFitModel.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Running predictions on test set...")
    y_true = test_df[label_col].tolist()
    y_pred = model.predict(test_df[text_col].tolist())

    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, unique_labels, OUTPUT_DIR)
    
    print("Generating classification report...")
    report = classification_report(y_true, y_pred, target_names=[str(l) for l in unique_labels])
    print("\n" + report)
    
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print(f"Classification report saved to {OUTPUT_DIR}/classification_report.txt")


if __name__ == "__main__":
    main()