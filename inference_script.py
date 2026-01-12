from setfit import SetFitModel
import pandas as pd
import torch
import os

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_DIR = "setfit_sentiment_model_safetensors"

def load_model():
    print(f"Loading SetFit model from {MODEL_DIR}...")
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Directory {MODEL_DIR} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SetFit automatically detects safetensors files if they exist in the folder
    model = SetFitModel.from_pretrained(MODEL_DIR)
    model.to(device)
    return model

def predict_sentiment(model, sentences):
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
    # 1. Load
    model = load_model()
    
    # 2. Dummy Data
    new_data = [
        "This feature is absolutely critical for my accounting workflow, please add it!",
        "The new update is okay, but it broke the invoice sorting.",
        "I hate the new UI, it's confusing and slow.",
    ]
    
    # 3. Predict
    df_results = predict_sentiment(model, new_data)
    
    print("\nResults:")
    print(df_results)