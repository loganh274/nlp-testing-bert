"""SetFit sentiment model training script."""
import os
import pandas as pd
import numpy as np
import torch

import json
import sys
import sklearn
import setfit
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gc
from typing import Dict, List, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl



# Configuration
TRAIN_DATA_PATH = "data/training_augmented.csv"
TEST_DATA_PATH = "data/test.csv"
MODEL_OUTPUT_DIR = "models/setfit_sentiment_model_safetensors"
CONFUSION_MATRIX_OUTPUT_DIR = "output/model_visualizations"
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
#BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 16
NUM_EPOCHS = (1, 16)

if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

def enhance_emotional_features(text):
    """Preserve and normalize emotional language features for better embedding."""
    # Normalize repeated punctuation (!!!! -> !! [EMPHASIS])
    text = re.sub(r'!{2,}', '!! [EMPHASIS]', text)
    text = re.sub(r'\?{2,}', '?? [QUESTION_EMPHASIS]', text)
    
    # Normalize repeated letters (sooooo -> so [ELONGATED])
    text = re.sub(r'(.)\1{2,}', r'\1\1 [ELONGATED]', text)
    
    # Mark ALL CAPS words (but don't lowercase them)
    def mark_caps(match):
        word = match.group(0)
        if len(word) > 2 and word.isupper():
            return f"{word} [CAPS]"
        return word
    text = re.sub(r'\b[A-Z]{2,}\b', mark_caps, text)
    
    return text


class MetricHistoryCallback(TrainerCallback):
    """Callback to capture training metrics at each logging step/epoch."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "embedding_loss": [],
            "eval_embedding_loss": [],
        }
        self.current_epoch = 0
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict] = None, **kwargs):
        """Capture metrics whenever they are logged."""
        if logs is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(value))
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Track epoch completion."""
        if state.epoch is not None:
            self.current_epoch = state.epoch
            if "epoch" not in self.history:
                self.history["epoch"] = []
            self.history["epoch"].append(float(state.epoch))
    
    def get_history(self) -> Dict[str, List[float]]:
        """Return the collected metric history."""
        return self.history


def get_model_stats(model: SetFitModel) -> Dict[str, Any]:
    """Compute model statistics including parameter count and size."""
    # Count parameters in the sentence transformer body
    body_params = 0
    body_trainable = 0
    if model.model_body is not None:
        body_params = sum(p.numel() for p in model.model_body.parameters())
        body_trainable = sum(p.numel() for p in model.model_body.parameters() if p.requires_grad)
    
    # Count parameters in the classification head
    head_params = 0
    head_trainable = 0
    if model.model_head is not None:
        if hasattr(model.model_head, 'parameters'):
            try:
                head_params = sum(p.numel() for p in model.model_head.parameters())  # type: ignore
                head_trainable = sum(p.numel() for p in model.model_head.parameters() if p.requires_grad)  # type: ignore
            except TypeError:
                # LogisticRegression head doesn't have PyTorch parameters
                pass
        if head_params == 0 and hasattr(model.model_head, 'coef_'):
            coef = getattr(model.model_head, 'coef_', None)
            intercept = getattr(model.model_head, 'intercept_', None)
            if coef is not None and intercept is not None:
                head_params = int(coef.size) + int(intercept.size)
                head_trainable = head_params
    
    total_params = body_params + head_params
    total_trainable = body_trainable + head_trainable
    
    # Estimate model size (assuming float32 for most params)
    model_size_mb = (body_params * 4) / (1024 * 1024)  
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": total_trainable,
        "body_parameters": body_params,
        "head_parameters": head_params,
        "model_size_mb": round(model_size_mb, 2),
        "model_description": f"SetFit model with {total_params:,} total parameters ({model_size_mb:.1f} MB)"
    }


def plot_training_history(history: Dict[str, List[float]], output_dir: str):
    """Generate and save training progress visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out non-numeric or empty entries
    plot_keys = [k for k, v in history.items() if v and all(isinstance(x, (int, float)) for x in v)]
    
    # Plot 1: Loss curves
    loss_keys = [k for k in plot_keys if 'loss' in k.lower()]
    if loss_keys:
        plt.figure(figsize=(10, 6))
        for key in loss_keys:
            values = history[key]
            epochs = list(range(1, len(values) + 1))
            label = key.replace('_', ' ').title()
            plt.plot(epochs, values, marker='o', label=label, linewidth=2, markersize=4)
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
        plt.close()
        print(f"Loss curve saved to {output_dir}/loss_curve.png")
    
    # Plot 2: Learning rate if available
    lr_keys = [k for k in plot_keys if 'learning_rate' in k.lower() or 'lr' in k.lower()]
    if lr_keys:
        plt.figure(figsize=(10, 6))
        for key in lr_keys:
            values = history[key]
            steps = list(range(1, len(values) + 1))
            plt.plot(steps, values, marker='.', linewidth=1)
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150)
        plt.close()
        print(f"Learning rate plot saved to {output_dir}/learning_rate.png")
    
    # Save history as JSON for later analysis
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


def plot_evaluation_metrics(y_true, y_pred, labels, output_dir: str) -> Dict[str, float]:
    """Generate comprehensive evaluation metrics and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
    }
    
    # Generate classification report
    report: Dict[str, Any] = classification_report(y_true, y_pred, target_names=[str(l) for l in labels], output_dict=True)  # type: ignore
    
    # Save classification report as JSON
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Classification report (JSON) saved to {report_path}")
    
    # Save classification report as text
    report_txt: str = classification_report(y_true, y_pred, target_names=[str(l) for l in labels], output_dict=False)  # type: ignore
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_txt)
    print(f"Classification report (TXT) saved to {output_dir}/classification_report.txt")
    
    # Plot: Metrics comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Overall metrics
    metric_names = ['Accuracy', 'F1 (Weighted)', 'Precision', 'Recall']
    metric_values = [metrics['accuracy'], metrics['f1_weighted'], 
                     metrics['precision_weighted'], metrics['recall_weighted']]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Overall Model Performance', fontsize=14)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: Per-class F1 scores
    class_f1 = [report[str(l)]['f1-score'] for l in labels]
    class_labels = [str(l) for l in labels]
    
    bars2 = axes[1].bar(class_labels, class_f1, color='#3498db', edgecolor='black', linewidth=1.2)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score by Class', fontsize=14)
    axes[1].axhline(y=metrics['f1_macro'], color='red', linestyle='--', alpha=0.7, label=f'Macro Avg: {metrics["f1_macro"]:.3f}')
    axes[1].legend()
    
    for bar, val in zip(bars2, class_f1):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=150)
    plt.close()
    print(f"Evaluation metrics chart saved to {output_dir}/evaluation_metrics.png")
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Final metrics saved to {metrics_path}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Generate and save confusion matrix visualization."""
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


def save_deployment_metadata(
    output_dir: str, 
    unique_labels: List, 
    model_stats: Optional[Dict] = None,
    training_history: Optional[Dict] = None,
    final_metrics: Optional[Dict] = None,
    training_samples: Optional[int] = None,
    test_samples: Optional[int] = None
):
    """Save comprehensive environment and model metadata for deployment and HuggingFace."""
    metadata = {
        "labels": [int(l) if isinstance(l, (np.integer, int)) else str(l) for l in unique_labels],
        "num_labels": len(unique_labels),
        "environment": {
            "python": sys.version,
            "scikit-learn": sklearn.__version__,
            "setfit": setfit.__version__,
            "torch": torch.__version__
        },
        "base_model": BASE_MODEL_NAME,
        "serialization": "safetensors",
        "training_config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "training_samples": training_samples,
            "test_samples": test_samples
        }
    }
    
    if model_stats:
        metadata["model_stats"] = model_stats
    
    if final_metrics:
        metadata["evaluation_metrics"] = final_metrics
    
    if training_history:
        # Include summary of training history
        metadata["training_summary"] = {
            "epochs_trained": len(training_history.get("epoch", [])),
            "final_loss": training_history.get("embedding_loss", [None])[-1] if training_history.get("embedding_loss") else None,
        }
    
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Deployment metadata saved to {output_dir}/model_metadata.json")
    
    return metadata


def get_col_names(df):
    """Detect text and label column names."""
    text = next((c for c in df.columns if 'text' in c.lower() or 'comment' in c.lower()), 'text')
    label = next((c for c in df.columns if 'label' in c.lower() or 'sentiment' in c.lower()), 'label')
    return text, label


def main():
    print("--- Starting SetFit Sentiment Model Training ---")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  
    else:
        device = "cpu"
    print(f"Training on {device.upper()}")

    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data file not found at {TRAIN_DATA_PATH}")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data file not found at {TEST_DATA_PATH}")

    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    print(f"Loading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)

    text_col, label_col = get_col_names(train_df)

    print("Enhancing emotional features in text...")
    train_df[text_col] = train_df[text_col].apply(enhance_emotional_features)
    test_df[text_col] = test_df[text_col].apply(enhance_emotional_features)
   
    train_df = train_df.dropna(subset=[text_col, label_col])
    test_df = test_df.dropna(subset=[text_col, label_col])

    unique_labels = sorted(train_df[label_col].unique().tolist())
    print(f"Training on {len(train_df)} rows. Found classes: {unique_labels}")
    print(f"Testing on {len(test_df)} rows.")

    train_ds = Dataset.from_pandas(train_df[[text_col, label_col]])
    test_ds = Dataset.from_pandas(test_df[[text_col, label_col]])

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = SetFitModel.from_pretrained(BASE_MODEL_NAME, labels=unique_labels)
    model.to(device)
    
    # Get and display model statistics
    model_stats = get_model_stats(model)
    print(f"\n--- Model Statistics ---")
    print(f"Total parameters: {model_stats['total_parameters']:,}")
    print(f"Trainable parameters: {model_stats['trainable_parameters']:,}")
    print(f"Model size: {model_stats['model_size_mb']} MB")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"------------------------\n")
    
    # Initialize metric tracking callback
    metric_callback = MetricHistoryCallback()

    print("Starting training...")
    args = TrainingArguments(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_iterations=10,
        loss=CosineSimilarityLoss,
        metric_for_best_model="embedding_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_strategy="epoch",
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        metric="accuracy",
        column_mapping={text_col: "text", label_col: "label"},
        callbacks=[metric_callback],
    )
    
    trainer.train()
    
    training_history = metric_callback.get_history()
    print(f"\nTraining history captured: {list(training_history.keys())}")

    print("Evaluating on test set...")
    eval_metrics = trainer.evaluate()
    print(f"Test Metrics: {eval_metrics}")
    
    preds = model.predict(test_df[text_col].tolist())
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CONFUSION_MATRIX_OUTPUT_DIR, exist_ok=True)
    
    print("\n--- Generating Visualizations ---")
    
    # 1. Training history plots (loss curves, learning rate)
    plot_training_history(training_history, CONFUSION_MATRIX_OUTPUT_DIR)
    
    # 2. Evaluation metrics (accuracy, F1, precision, recall charts)
    final_metrics = plot_evaluation_metrics(
        test_df[label_col], preds, unique_labels, CONFUSION_MATRIX_OUTPUT_DIR
    )
    
    # 3. Confusion matrix
    plot_confusion_matrix(test_df[label_col], preds, unique_labels, CONFUSION_MATRIX_OUTPUT_DIR)
    
    print("\n--- Final Evaluation Results ---")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1 (Weighted): {final_metrics['f1_weighted']:.4f}")
    print(f"F1 (Macro): {final_metrics['f1_macro']:.4f}")
    print(f"Precision (Weighted): {final_metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {final_metrics['recall_weighted']:.4f}")
    print("--------------------------------\n")

    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR, safe_serialization=True)
    
    # Save comprehensive deployment metadata
    save_deployment_metadata(
        MODEL_OUTPUT_DIR, 
        unique_labels,
        model_stats=model_stats,
        training_history=training_history,
        final_metrics=final_metrics,
        training_samples=len(train_df),
        test_samples=len(test_df)
    )
    
    print("Done! Model saved in safetensors format with full metadata.")


if __name__ == "__main__":
    main()