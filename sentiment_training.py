import pandas as pd
import numpy as np
import torch
import os
import json
import sys
import sklearn
import setfit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from setfit import SetFitModel, Trainer, TrainingArguments # Changed SetFitTrainer to Trainer
from datasets import Dataset

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATH = "xpi_labeled_data_augmented.csv"
MODEL_OUTPUT_DIR = "setfit_sentiment_model_safetensors"

# "all-mpnet-base-v2" is widely considered the best general-purpose model 
BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 

# Training Params
NUM_EPOCHS = 3
BATCH_SIZE = 16

def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('SetFit Sentiment Classification Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    plt.close()

def save_deployment_metadata(output_dir, unique_labels):
    """
    Saves environment details. Crucial for Snowflake deployment.
    """
    metadata = {
        "labels": [int(l) if isinstance(l, (np.integer, int)) else str(l) for l in unique_labels],
        "environment": {
            "python": sys.version,
            "scikit-learn": sklearn.__version__,
            "setfit": setfit.__version__,
            "torch": torch.__version__
        },
        "base_model": BASE_MODEL_NAME,
        "serialization": "safetensors"
    }
    
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Deployment metadata saved to {output_dir}/model_metadata.json")



def main():
    print("--- Starting SetFit Sentiment Model Training (Safetensors) ---")

    # 1. HARDWARE CHECK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Check: Training on {device.upper()}")

    # 2. LOAD DATA
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    text_col = next((col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()), 'text')
    label_col = next((col for col in df.columns if 'label' in col.lower() or 'sentiment' in col.lower()), 'label')
    
    df = df.dropna(subset=[text_col, label_col])
    
    unique_labels = sorted(list(df[label_col].unique()))
    print(f"Found {len(unique_labels)} unique classes: {unique_labels}")

    # Split Data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    train_ds = Dataset.from_pandas(train_df[[text_col, label_col]])
    test_ds = Dataset.from_pandas(test_df[[text_col, label_col]])

    # 3. INITIALIZE SETFIT MODEL
    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = SetFitModel.from_pretrained(
        BASE_MODEL_NAME,
        labels=unique_labels,
    )
    model.to(device)

    # 4. TRAIN (UPDATED FOR SETFIT v1.0+)
    print("Starting training...")
    
    # Define training arguments separately
    args = TrainingArguments(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch", # Optional: evaluate at end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_iterations=20,
        loss="CosineSimilarityLoss", # "loss_class" is now "loss" (can be string or class)
    )

    trainer = Trainer(
        model=model,
        args=args,                  # Pass arguments here
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="accuracy",
        column_mapping={text_col: "text", label_col: "label"}
    )
    
    trainer.train()

    # 5. EVALUATION
    print("Evaluating on Test Set...")
    metrics = trainer.evaluate()
    print(f"Test Metrics: {metrics}")
    
    preds = model.predict(test_df[text_col].tolist())
    
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    plot_confusion_matrix(test_df[label_col], preds, unique_labels, MODEL_OUTPUT_DIR)

    # 6. SAVE MODEL
    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR, safe_serialization=True)
    save_deployment_metadata(MODEL_OUTPUT_DIR, unique_labels)
    print("Done! Model saved in safetensors format.")

if __name__ == "__main__":
    main()