"""Local inference script for SetFit sentiment model."""

from setfit import SetFitModel
import pandas as pd
import torch
import os

MODEL_DIR = "../models/setfit_sentiment_model_safetensors"


def load_model():
    """Load the SetFit model from local directory."""
    print(f"Loading SetFit model from {MODEL_DIR}...")
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Directory {MODEL_DIR} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SetFitModel.from_pretrained(MODEL_DIR)
    model.to(device)
    return model


def predict_sentiment(model, sentences):
    """Run sentiment prediction on a list of sentences."""
    print(f"Predicting on {len(sentences)} sentences...")
    preds = model.predict(sentences)
    probas = model.predict_proba(sentences)
    
    results = []
    for i, sentence in enumerate(sentences):
        results.append({
            "text": sentence,
            "predicted_label": preds[i].item(),
            "confidence_scores": probas[i].tolist()
        })
        
    return pd.DataFrame(results)


def run_inference(input_csv, output_folder="sentiment_output", max_rows=None, batch_size=100):
    """
    Run sentiment inference on a CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_folder: Folder to save output (default: sentiment_output)
        max_rows: Maximum number of rows to process (None = all rows)
        batch_size: Number of comments to process at a time (default: 100)
    
    Returns:
        DataFrame with SENTIMENT_SCORE column added
    """
    os.makedirs(output_folder, exist_ok=True)
    model = load_model()
    
    print(f"Loading CSV file: {input_csv}...")
    if max_rows is not None:
        df = pd.read_csv(input_csv, nrows=max_rows)
        print(f"Processing first {max_rows} rows only (for testing)")
    else:
        df = pd.read_csv(input_csv)
    
    print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    COMMENT_COLUMN_INDEX = 4
    comment_column_name = df.columns[COMMENT_COLUMN_INDEX]
    print(f"Analyzing column: '{comment_column_name}'")
    
    comments = df[comment_column_name].fillna("").astype(str).tolist()
    
    print(f"Running sentiment analysis on {len(comments)} comments...")
    all_predictions = []
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(comments) + batch_size - 1) // batch_size
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        preds = model.predict(batch)
        all_predictions.extend([p.item() if hasattr(p, 'item') else p for p in preds])
    
    df["SENTIMENT_SCORE"] = all_predictions
    
    input_basename = os.path.splitext(os.path.basename(input_csv))[0]
    if max_rows is not None:
        output_filename = f"{input_basename}_sentiment_{max_rows}rows.csv"
    else:
        output_filename = f"{input_basename}_sentiment.csv"
    output_path = os.path.join(output_folder, output_filename)
    
    print(f"Saving results to: {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"\nDone! Added SENTIMENT_SCORE column")
    print(f"Output saved to: {output_path}")
    print(f"Sentiment distribution:")
    print(df["SENTIMENT_SCORE"].value_counts())
    
    return df


if __name__ == "__main__":
    # === CONFIGURATION ===
    INPUT_CSV = "../data/All XPI Comments filtered (2).csv"
    OUTPUT_FOLDER = "../output/sentiment_output"
    MAX_ROWS = 1000  # Set to None to process all rows
    BATCH_SIZE = 100
    # =====================
    
    run_inference(
        input_csv=INPUT_CSV,
        output_folder=OUTPUT_FOLDER,
        max_rows=MAX_ROWS,
        batch_size=BATCH_SIZE
    )