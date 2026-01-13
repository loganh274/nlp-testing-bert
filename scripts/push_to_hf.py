"""Upload trained model to Hugging Face Hub with model card and visualizations."""

import os
import json
from huggingface_hub import login, upload_file, HfApi
from setfit import SetFitModel

# Configuration
MODEL_DIR = "models/setfit_sentiment_model_safetensors"
VISUALIZATION_DIR = "output/model_visualizations"
REPO_ID = "loganh274/nlp-testing-setfit"


def load_metadata(model_dir: str) -> dict:
    """Load model metadata from saved JSON file."""
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def load_training_history(viz_dir: str) -> dict:
    """Load training history from visualization directory."""
    history_path = os.path.join(viz_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    return {}


def load_final_metrics(viz_dir: str) -> dict:
    """Load final metrics from visualization directory."""
    metrics_path = os.path.join(viz_dir, "final_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


def load_classification_report(viz_dir: str) -> str:
    """Load classification report text."""
    report_path = os.path.join(viz_dir, "classification_report.txt")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return f.read()
    return ""


def generate_model_card(metadata: dict, training_history: dict, final_metrics: dict, classification_report: str) -> str:
    """Generate a comprehensive model card for Hugging Face."""
    model_stats = metadata.get("model_stats", {})
    eval_metrics = metadata.get("evaluation_metrics", {})
    training_config = metadata.get("training_config", {})
    training_summary = metadata.get("training_summary", {})
    env = metadata.get("environment", {})
    
    # Override with more detailed metrics if available
    if final_metrics:
        eval_metrics.update(final_metrics)
    
    # Helper function to format metric values
    def fmt_metric(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value) if value is not None else "N/A"
    
    def fmt_number(value):
        if isinstance(value, (int, float)):
            return f"{value:,}"
        return str(value) if value is not None else "N/A"

    # Build training history summary
    training_details = ""
    if training_history:
        epochs = training_history.get("epoch", [])
        losses = training_history.get("embedding_loss", [])
        eval_losses = training_history.get("eval_embedding_loss", [])
        
        if epochs or losses:
            training_details = "\n### Training Progress\n\n"
            if losses:
                training_details += f"- **Initial Loss:** {fmt_metric(losses[0] if losses else None)}\n"
                training_details += f"- **Final Loss:** {fmt_metric(losses[-1] if losses else None)}\n"
            if eval_losses:
                training_details += f"- **Eval Loss:** {fmt_metric(eval_losses[-1] if eval_losses else None)}\n"
            
            # Runtime stats
            runtime = training_history.get("train_runtime", [])
            samples_per_sec = training_history.get("train_samples_per_second", [])
            if runtime:
                training_details += f"- **Training Runtime:** {fmt_metric(runtime[-1])} seconds\n"
            if samples_per_sec:
                training_details += f"- **Samples/Second:** {fmt_metric(samples_per_sec[-1])}\n"

    # Build per-class metrics if available
    per_class_section = ""
    if classification_report:
        per_class_section = f"""
### Per-Class Performance

```
{classification_report}
```
"""

    card = f"""---
language:
- en
license: apache-2.0
library_name: setfit
tags:
- setfit
- sentence-transformers
- text-classification
- sentiment-analysis
- few-shot-learning
pipeline_tag: text-classification
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: SetFit Sentiment Analysis
  results:
  - task:
      type: text-classification
      name: Sentiment Analysis
    metrics:
    - name: Accuracy
      type: accuracy
      value: {eval_metrics.get('accuracy', 'N/A')}
    - name: F1 (Weighted)
      type: f1
      value: {eval_metrics.get('f1_weighted', 'N/A')}
    - name: Precision (Weighted)
      type: precision
      value: {eval_metrics.get('precision_weighted', 'N/A')}
    - name: Recall (Weighted)
      type: recall
      value: {eval_metrics.get('recall_weighted', 'N/A')}
---

# SetFit Sentiment Analysis Model

This is a [SetFit](https://github.com/huggingface/setfit) model fine-tuned for sentiment classification on customer feedback data.

## Model Description

| Property | Value |
|----------|-------|
| **Base Model** | [{metadata.get('base_model', 'N/A')}](https://huggingface.co/{metadata.get('base_model', '')}) |
| **Total Parameters** | {fmt_number(model_stats.get('total_parameters'))} |
| **Trainable Parameters** | {fmt_number(model_stats.get('trainable_parameters'))} |
| **Body Parameters** | {fmt_number(model_stats.get('body_parameters'))} |
| **Head Parameters** | {fmt_number(model_stats.get('head_parameters'))} |
| **Model Size** | {model_stats.get('model_size_mb', 'N/A')} MB |
| **Labels** | {metadata.get('labels', [])} |
| **Number of Classes** | {metadata.get('num_labels', 'N/A')} |
| **Serialization** | {metadata.get('serialization', 'safetensors')} |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | {training_config.get('batch_size', 'N/A')} |
| **Epochs** | {training_config.get('num_epochs', 'N/A')} |
| **Training Samples** | {fmt_number(training_config.get('training_samples'))} |
| **Test Samples** | {fmt_number(training_config.get('test_samples'))} |
| **Loss Function** | CosineSimilarityLoss |
| **Metric for Best Model** | embedding_loss |
{training_details}
## Evaluation Results

| Metric | Score |
|--------|-------|
| **Accuracy** | {fmt_metric(eval_metrics.get('accuracy'))} |
| **F1 (Weighted)** | {fmt_metric(eval_metrics.get('f1_weighted'))} |
| **F1 (Macro)** | {fmt_metric(eval_metrics.get('f1_macro'))} |
| **Precision (Weighted)** | {fmt_metric(eval_metrics.get('precision_weighted'))} |
| **Precision (Macro)** | {fmt_metric(eval_metrics.get('precision_macro'))} |
| **Recall (Weighted)** | {fmt_metric(eval_metrics.get('recall_weighted'))} |
| **Recall (Macro)** | {fmt_metric(eval_metrics.get('recall_macro'))} |
{per_class_section}
## Visualizations

### Evaluation Metrics Overview
<p align="center">
  <img src="evaluation_metrics.png" alt="Evaluation Metrics" width="800"/>
</p>

### Confusion Matrix
<p align="center">
  <img src="confusion_matrix.png" alt="Confusion Matrix" width="600"/>
</p>

### Training Loss Curve
<p align="center">
  <img src="loss_curve.png" alt="Training Loss Curve" width="600"/>
</p>

### Learning Rate Schedule
<p align="center">
  <img src="learning_rate.png" alt="Learning Rate Schedule" width="600"/>
</p>

## Usage

```python
from setfit import SetFitModel

# Load the model
model = SetFitModel.from_pretrained("{REPO_ID}")

# Single prediction
text = "This product exceeded my expectations!"
prediction = model.predict([text])
print(f"Sentiment: {{prediction[0]}}")

# Batch prediction
texts = [
    "Amazing quality, highly recommend!",
    "It's okay, nothing special.",
    "Terrible experience, very disappointed.",
]
predictions = model.predict(texts)
probabilities = model.predict_proba(texts)

for text, pred, prob in zip(texts, predictions, probabilities):
    print(f"Text: {{text}}")
    print(f"  Prediction: {{pred}}, Confidence: {{max(prob):.2%}}")
```

## Label Mapping

| Label | Sentiment |
|-------|-----------|
| 0 | Negative |
| 1 | Somewhat Negative |
| 2 | Neutral |
| 3 | Somewhat Positive |
| 4 | Positive |

## Environment

| Package | Version |
|---------|---------|
| Python | {env.get('python', 'N/A').split()[0] if env.get('python') else 'N/A'} |
| SetFit | {env.get('setfit', 'N/A')} |
| PyTorch | {env.get('torch', 'N/A')} |
| scikit-learn | {env.get('scikit-learn', 'N/A')} |
| Transformers | {env.get('transformers', 'N/A') if 'transformers' in env else 'N/A'} |

## Citation

If you use this model, please cite the SetFit paper:

```bibtex
@article{{tunstall2022efficient,
  title={{Efficient Few-Shot Learning Without Prompts}},
  author={{Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren}},
  journal={{arXiv preprint arXiv:2209.11055}},
  year={{2022}}
}}
```

## License

Apache 2.0
"""
    return card


def upload_visualizations(api: HfApi, repo_id: str, viz_dir: str):
    """Upload visualization images to the Hub repository."""
    viz_files = [
        "confusion_matrix.png",
        "evaluation_metrics.png", 
        "loss_curve.png",
        "learning_rate.png",
        "classification_report.txt",
        "classification_report.json",
        "final_metrics.json",
        "training_history.json",
    ]
    
    uploaded = []
    for filename in viz_files:
        filepath = os.path.join(viz_dir, filename)
        if os.path.exists(filepath):
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            uploaded.append(filename)
        else:
            print(f"  Skipping {filename} (not found)")
    
    return uploaded


def main():
    print("=" * 60)
    print("Uploading Model to Hugging Face Hub")
    print("=" * 60)
    
    # Login to Hugging Face
    print("\n[1/7] Authenticating with Hugging Face...")
    login()
    
    # Initialize API
    api = HfApi()
    
    # Load model and all metadata
    print(f"\n[2/7] Loading model from {MODEL_DIR}...")
    model = SetFitModel.from_pretrained(MODEL_DIR)
    
    print(f"\n[3/7] Loading metadata and training history...")
    metadata = load_metadata(MODEL_DIR)
    training_history = load_training_history(VISUALIZATION_DIR)
    final_metrics = load_final_metrics(VISUALIZATION_DIR)
    classification_report = load_classification_report(VISUALIZATION_DIR)
    
    print(f"  - Metadata keys: {list(metadata.keys())}")
    print(f"  - Training history keys: {list(training_history.keys())}")
    print(f"  - Final metrics: {list(final_metrics.keys()) if final_metrics else 'None'}")
    print(f"  - Classification report: {'Loaded' if classification_report else 'Not found'}")
    
    # Generate model card
    print(f"\n[4/7] Generating model card...")
    model_card = generate_model_card(metadata, training_history, final_metrics, classification_report)
    readme_path = os.path.join(MODEL_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)
    print(f"  Model card saved to {readme_path}")
    
    # Push model to hub first (this will create its own README)
    print(f"\n[5/7] Pushing model to {REPO_ID}...")
    model.push_to_hub(REPO_ID)
    
    # Upload visualizations
    print(f"\n[6/7] Uploading visualizations from {VISUALIZATION_DIR}...")
    uploaded = upload_visualizations(api, REPO_ID, VISUALIZATION_DIR)
    
    # Upload additional metadata files from model dir
    metadata_files = ["model_metadata.json"]
    for filename in metadata_files:
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
            )
    
    # Upload custom README.md AFTER model push to overwrite the auto-generated one
    print(f"\n[7/7] Uploading custom README.md (overwriting auto-generated)...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("  Custom model card uploaded successfully!")
    
    print("\n" + "=" * 60)
    print(f"✅ Model successfully uploaded to:")
    print(f"   https://huggingface.co/{REPO_ID}")
    print("=" * 60)
    print(f"\nUploaded {len(uploaded)} visualization files")
    print("Done!")


if __name__ == "__main__":
    main()