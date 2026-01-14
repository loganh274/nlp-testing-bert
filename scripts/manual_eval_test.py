"""
Standalone evaluation script for pretrained SetFit sentiment model.
Usage: python evaluate_model.py --model_path <path_to_model> --data_path <path_to_test_data>
"""

import argparse
import json
import pandas as pd
from setfit import SetFitModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_model(model_path: str) -> SetFitModel:
    """Load a pretrained SetFit model."""
    print(f"Loading model from: {model_path}")
    model = SetFitModel.from_pretrained(model_path)
    return model


def load_test_data(data_path: str, text_column: str = "text", label_column: str = "label"):
    """Load test data from CSV."""
    print(f"Loading test data from: {data_path}")
    df = pd.read_csv(data_path)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    return texts, labels


def evaluate(model: SetFitModel, texts: list, true_labels: list, output_path: str | None = None):
    """Run evaluation and print metrics."""
    print("Running predictions...")
    predictions = model.predict(texts)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average="macro")
    f1_weighted = f1_score(true_labels, predictions, average="weighted")
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"F1 (macro):         {f1_macro:.4f}")
    print(f"F1 (weighted):      {f1_weighted:.4f}")
    print(f"Precision (macro):  {precision:.4f}")
    print(f"Recall (macro):     {recall:.4f}")
    print("\n" + "-" * 50)
    print("Classification Report:")
    print("-" * 50)
    report = classification_report(true_labels, predictions)
    print(report)
    
    print("\nConfusion Matrix:")
    print("-" * 50)
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    
    # Save results if output path provided
    if output_path:
        results = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision,
            "recall_macro": recall,
            "classification_report": classification_report(true_labels, predictions, output_dict=True),
            "confusion_matrix": cm.tolist(),
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained SetFit model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--text_column", type=str, default="text", help="Name of text column")
    parser.add_argument("--label_column", type=str, default="label", help="Name of label column")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results JSON")
    
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    texts, labels = load_test_data(args.data_path, args.text_column, args.label_column)
    evaluate(model, texts, labels, args.output_path)


if __name__ == "__main__":
    main()