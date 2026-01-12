"""Snowflake ML Functions inference script for SetFit sentiment model."""

import os
from setfit import SetFitModel

MODEL_ID = "loganh274/nlp-testing-setfit"
CACHE_DIR = "/tmp/hf_cache"

_model = None


def get_model():
    """Load and cache the model using singleton pattern for Snowflake efficiency."""
    global _model
    if _model is None:
        os.environ["HF_HOME"] = CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
        
        print(f"Downloading model from Hugging Face: {MODEL_ID}")
        _model = SetFitModel.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        print("Model loaded successfully!")
    return _model


def predict_sentiment(text: str) -> int:
    """Predict sentiment for a single text string. Returns integer label."""
    model = get_model()
    prediction = model.predict([text])
    return int(prediction[0])


def predict_sentiment_batch(texts: list) -> list:
    """Predict sentiment for a batch of texts. Returns list of integer labels."""
    model = get_model()
    predictions = model.predict(texts)
    return [int(p) for p in predictions]


def predict_with_confidence(text: str) -> dict:
    """Predict sentiment with confidence scores. Returns dict with label and scores."""
    model = get_model()
    prediction = model.predict([text])
    probas = model.predict_proba([text])
    
    return {
        "label": int(prediction[0]),
        "confidence_scores": probas[0].tolist()
    }


if __name__ == "__main__":
    test_texts = [
        "This feature is absolutely critical for my accounting workflow!",
        "The new update is okay, but it broke the invoice sorting.",
        "I hate the new UI, it's confusing and slow.",
    ]
    
    print("Testing model download and inference...")
    for text in test_texts:
        result = predict_with_confidence(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence_scores']}")
