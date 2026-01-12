"""Local inference script for SetFit sentiment model."""

from setfit import SetFitModel
import pandas as pd
import torch
import os

MODEL_DIR = "setfit_sentiment_model_safetensors"


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


if __name__ == "__main__":
    model = load_model()
    
    test_data = [
        "This feature is absolutely critical for my accounting workflow, please add it!",
        "The new update is okay, but it broke the invoice sorting.",
        "I hate the new UI, it's confusing and slow.",
    ]
    
    df_results = predict_sentiment(model, test_data)
    
    print("\nResults:")
    print(df_results)