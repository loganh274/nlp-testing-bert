"""Upload trained model to Hugging Face Hub."""

from huggingface_hub import login, upload_folder

login()

upload_folder(
    folder_path="setfit_sentiment_model_safetensors",
    repo_id="loganh274/nlp-testing-setfit",
    repo_type="model"
)
