"""Analyze misclassified examples to understand model weaknesses."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from setfit import SetFitModel
from sklearn.metrics import confusion_matrix
import os

# Configuration
MODEL_PATH = "models/setfit_sentiment_model_safetensors"
TEST_DATA_PATH = "data/test.csv"
OUTPUT_DIR = "output/model_visualizations"

# Label mapping (adjust if your labels differ)
LABEL_NAMES = {
    0: "Very Negative",
    1: "Negative", 
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}


def load_test_data(path: str) -> pd.DataFrame:
    """Load test data and detect column names."""
    df = pd.read_csv(path)
    
    # Detect text column
    text_col = None
    for col in ['text', 'Text', 'comment', 'Comment', 'sentence']:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]
    
    # Detect label column
    label_col = None
    for col in ['label', 'Label', 'sentiment', 'Sentiment']:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[1]
    
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    return df[['text', 'label']]


def analyze_confusion_matrix(y_true, y_pred, labels):
    """Analyze the confusion matrix to identify problem areas."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    # Print confusion matrix with labels
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("-" * 50)
    
    # Header
    header = "Actual\\Pred |"
    for label in labels:
        header += f" {LABEL_NAMES.get(label, label):^12} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, label in enumerate(labels):
        row = f"{LABEL_NAMES.get(label, label):>11} |"
        for j in range(len(labels)):
            val = cm[i, j]
            if i == j:
                row += f" \033[92m{val:^12}\033[0m |"  # Green for correct
            elif val > 0:
                row += f" \033[91m{val:^12}\033[0m |"  # Red for errors
            else:
                row += f" {val:^12} |"
        print(row)
    
    print("\n" + "-"*50)
    print("KEY CONFUSION PAIRS (where errors occur):")
    print("-"*50)
    
    # Find significant confusion pairs
    confusion_pairs = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'actual': actual,
                    'actual_name': LABEL_NAMES.get(actual, actual),
                    'predicted': predicted,
                    'predicted_name': LABEL_NAMES.get(predicted, predicted),
                    'count': cm[i, j],
                    'pct_of_actual': cm[i, j] / cm[i].sum() * 100
                })
    
    # Sort by count
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    for pair in confusion_pairs[:10]:  # Top 10 confusion pairs
        print(f"  {pair['actual_name']:>14} → {pair['predicted_name']:<14}: "
              f"{pair['count']:2} errors ({pair['pct_of_actual']:.0f}% of {pair['actual_name']})")
    
    return cm, confusion_pairs


def get_misclassified_examples(df: pd.DataFrame, y_pred, labels) -> pd.DataFrame:
    """Extract all misclassified examples with details."""
    df = df.copy()
    df['predicted'] = y_pred
    df['predicted_name'] = df['predicted'].map(LABEL_NAMES)
    df['actual_name'] = df['label'].map(LABEL_NAMES)
    df['correct'] = df['label'] == df['predicted']
    
    misclassified = df[~df['correct']].copy()
    misclassified['error_type'] = misclassified.apply(
        lambda r: f"{r['actual_name']} → {r['predicted_name']}", axis=1
    )
    
    return misclassified


def print_misclassified_examples(misclassified: pd.DataFrame, confusion_pairs: list):
    """Print misclassified examples grouped by error type."""
    
    print("\n" + "="*60)
    print("MISCLASSIFIED EXAMPLES BY ERROR TYPE")
    print("="*60)
    
    for pair in confusion_pairs[:5]:  # Top 5 error types
        error_type = f"{pair['actual_name']} → {pair['predicted_name']}"
        examples = misclassified[misclassified['error_type'] == error_type]
        
        print(f"\n{'─'*60}")
        print(f"❌ {error_type} ({len(examples)} errors)")
        print(f"{'─'*60}")
        
        for idx, row in examples.head(5).iterrows():  # Show up to 5 examples
            text = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            print(f"  • \"{text}\"")
        
        if len(examples) > 5:
            print(f"  ... and {len(examples) - 5} more")


def plot_enhanced_confusion_matrix(cm, labels, output_dir):
    """Create an enhanced confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    
    # Normalize for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation labels showing count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.0f}%)"
    
    label_names = [LABEL_NAMES.get(l, str(l)) for l in labels]
    
    sns.heatmap(
        cm, 
        annot=annot,
        fmt='',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.title('Confusion Matrix with Counts and Percentages', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confusion_matrix_detailed.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved detailed confusion matrix to: {output_path}")


def save_misclassification_report(misclassified: pd.DataFrame, confusion_pairs: list, output_dir: str):
    """Save misclassification analysis to files."""
    
    # Save CSV of all misclassified examples
    csv_path = os.path.join(output_dir, 'misclassified_examples.csv')
    misclassified.to_csv(csv_path, index=False)
    print(f"✅ Saved misclassified examples to: {csv_path}")
    
    # Save JSON summary
    summary = {
        'total_misclassified': int(len(misclassified)),  # Convert to Python int
        'error_breakdown': [
            {
                'error_type': f"{p['actual_name']} → {p['predicted_name']}",
                'count': int(p['count']),  # Convert to Python int
                'percentage_of_class': round(float(p['pct_of_actual']), 1)  # Convert to Python float
            }
            for p in confusion_pairs
        ],
        'recommendations': generate_recommendations(confusion_pairs)
    }
    
    json_path = os.path.join(output_dir, 'misclassification_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved analysis summary to: {json_path}")


def generate_recommendations(confusion_pairs: list) -> list:
    """Generate actionable recommendations based on confusion patterns."""
    recommendations = []
    
    for pair in confusion_pairs[:3]:  # Focus on top 3 issues
        actual = pair['actual_name']
        predicted = pair['predicted_name']
        count = pair['count']
        pct = pair['pct_of_actual']
        
        if pct >= 20:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f"{actual} frequently misclassified as {predicted} ({pct:.0f}%)",
                'action': f"Add more training examples that clearly distinguish {actual} from {predicted}"
            })
        
        # Check for adjacent class confusion (sentiment scale)
        actual_idx = list(LABEL_NAMES.values()).index(actual)
        pred_idx = list(LABEL_NAMES.values()).index(predicted)
        
        if abs(actual_idx - pred_idx) == 1:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': f"Adjacent class confusion: {actual} ↔ {predicted}",
                'action': "Consider adding examples at the boundary between these classes, or merge them if distinction isn't critical"
            })
    
    return recommendations


def main():
    print("Loading model...")
    model = SetFitModel.from_pretrained(MODEL_PATH)
    
    print("Loading test data...")
    df = load_test_data(TEST_DATA_PATH)
    
    print(f"Running predictions on {len(df)} test samples...")
    y_pred = model.predict(df['text'].tolist())
    y_true = df['label'].values
    
    labels = sorted(df['label'].unique())
    
    # Analyze confusion matrix
    cm, confusion_pairs = analyze_confusion_matrix(y_true, y_pred, labels)
    
    # Get misclassified examples
    misclassified = get_misclassified_examples(df, y_pred, labels)
    
    # Print examples
    print_misclassified_examples(misclassified, confusion_pairs)
    
    # Create enhanced visualization
    plot_enhanced_confusion_matrix(cm, labels, OUTPUT_DIR)
    
    # Save reports
    save_misclassification_report(misclassified, confusion_pairs, OUTPUT_DIR)
    
    # Print recommendations
    recommendations = generate_recommendations(confusion_pairs)
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    for rec in recommendations:
        print(f"\n[{rec['priority']}] {rec['issue']}")
        print(f"   → {rec['action']}")


if __name__ == "__main__":
    main()