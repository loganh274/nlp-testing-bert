"""Merge original and synthetic training data with deduplication."""

import pandas as pd
import os

ORIGINAL_TRAINING_PATH = "data/training.csv"
SYNTHETIC_PATH = "data/synthetic_training_examples.csv"
OUTPUT_PATH = "data/training_augmented.csv"


def main():
    print("="*50)
    print("MERGING TRAINING DATA")
    print("="*50)
    
    # Load original training data
    print("\nLoading original training data...")
    original_df = pd.read_csv(ORIGINAL_TRAINING_PATH)
    
    # Ensure column names are standardized
    if 'text' not in original_df.columns:
        original_df = original_df.rename(columns={original_df.columns[0]: 'text', original_df.columns[1]: 'label'})
    
    original_df = original_df[['text', 'label']].copy()
    
    print(f"  Total examples: {len(original_df)}")
    print(f"  Distribution:")
    label_names = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    for label in sorted(original_df['label'].unique()):
        count = len(original_df[original_df['label'] == label])
        print(f"    {label} ({label_names.get(label, '?'):>13}): {count}")
    
    # Load synthetic data
    print("\nLoading synthetic training data...")
    if not os.path.exists(SYNTHETIC_PATH):
        print(f"  ❌ Synthetic data not found at {SYNTHETIC_PATH}")
        print("  Run generate_synthetic_training_data.py first")
        return
    
    synthetic_df = pd.read_csv(SYNTHETIC_PATH)
    synthetic_df = synthetic_df[['text', 'label']].copy()
    
    print(f"  Total examples: {len(synthetic_df)}")
    print(f"  Distribution:")
    for label in sorted(synthetic_df['label'].unique()):
        count = len(synthetic_df[synthetic_df['label'] == label])
        print(f"    {label} ({label_names.get(label, '?'):>13}): {count}")
    
    # Only remove duplicates from synthetic that exist in original
    # (Never remove original data)
    print("\nFiltering synthetic duplicates...")
    original_texts = set(original_df['text'].str.strip().str.lower())
    
    synthetic_before = len(synthetic_df)
    synthetic_df['text_normalized'] = synthetic_df['text'].str.strip().str.lower()
    synthetic_df = synthetic_df[~synthetic_df['text_normalized'].isin(original_texts)]
    synthetic_df = synthetic_df.drop(columns=['text_normalized'])
    
    duplicates_removed = synthetic_before - len(synthetic_df)
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} synthetic examples that duplicated original data")
    
    # Merge - original first, then filtered synthetic
    print("\nMerging datasets...")
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    # Save
    combined_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Saved to {OUTPUT_PATH}")
    print(f"\n" + "="*50)
    print("FINAL DATASET")
    print("="*50)
    print(f"  Total examples: {len(combined_df)}")
    print(f"  Distribution:")
    for label in sorted(combined_df['label'].unique()):
        count = len(combined_df[combined_df['label'] == label])
        pct = count / len(combined_df) * 100
        print(f"    {label} ({label_names.get(label, '?'):>13}): {count:4} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()