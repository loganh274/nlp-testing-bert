# SetFit Sentiment Analysis Model

A fine-tuned [SetFit](https://github.com/huggingface/setfit) model for sentiment classification, trained on XPI customer feedback data.

**Model on Hugging Face:** [loganh274/nlp-testing-setfit](https://huggingface.co/loganh274/nlp-testing-setfit)

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/loganh274/nlp-testing-bert.git
cd nlp-testing-bert

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Inference (Download from Hugging Face)

```bash
python scripts/snowflake_inference.py
```

This will automatically download the pretrained model from Hugging Face and run predictions on sample texts.

### Use in Your Code

```python
from setfit import SetFitModel

# Download from Hugging Face
model = SetFitModel.from_pretrained("loganh274/nlp-testing-setfit")

# Predict
texts = ["This feature is great!", "I hate the new update."]
predictions = model.predict(texts)
print(predictions)  # [2, 0] (example output)
```

---

## Project Structure

```
nlp-testing-bert/
├── scripts/           # Python scripts for training, inference, and utilities
├── data/              # Training data and input CSV files
├── models/            # Trained model output (setfit_sentiment_model_safetensors/)
├── output/            # Inference results and visualizations
├── checkpoints/       # Training checkpoints
└── requirements.txt   # Python dependencies
```

### Scripts

| File | Description |
|------|-------------|
| `scripts/sentiment_training.py` | Training script for fine-tuning the model |
| `scripts/inference_script.py` | CSV batch inference with configurable row limits |
| `scripts/snowflake_inference.py` | Snowflake ML-ready inference script |
| `scripts/push_to_hf.py` | Script to upload model to Hugging Face |
| `scripts/generate_graphs.py` | Visualizations for model performance |

---

## CSV Batch Inference

Analyze sentiment on CSV files and output results with a new `SENTIMENT_SCORE` column.

### Configuration

Edit the settings at the bottom of `scripts/inference_script.py`:

```python
# ==================== CONFIGURATION ====================
INPUT_CSV = "../data/All XPI Comments filtered (2).csv"
OUTPUT_FOLDER = "../output/sentiment_output"

# Set MAX_ROWS to limit processing for testing (e.g., 1000, 5000)
# Set to None to process ALL rows
MAX_ROWS = 1000  # Change this to None to process all rows

BATCH_SIZE = 100  # Number of comments per batch
# =======================================================
```

### Run Inference

```bash
python scripts/inference_script.py
```

### Output

- Results are saved to `output/sentiment_output/` folder
- Output filename includes row count when limited (e.g., `*_sentiment_1000rows.csv`)
- The 5th column (`COMMENT_BODY`) is analyzed and `SENTIMENT_SCORE` is added as the 7th column

### Testing First

1. Set `MAX_ROWS = 1000` (or 5000) to test on a subset
2. Verify the output in `output/sentiment_output/`
3. Set `MAX_ROWS = None` to process all rows

---

## Training Your Own Model

1. Prepare your labeled data as a CSV with `text` and `label` columns
2. Place it in the `data/` folder and update `DATA_PATH` in `scripts/sentiment_training.py`
3. Run training:

```bash
python scripts/sentiment_training.py
```

The model will be saved to `models/setfit_sentiment_model_safetensors/`.

---

## Snowflake ML Deployment

Deploy the sentiment model as a Snowflake User-Defined Function (UDF).

### Step 1: Upload Script to Snowflake Stage

First, create a stage and upload the inference script:

```sql
-- Create a stage for Python files
CREATE STAGE IF NOT EXISTS ml_models_stage;

-- Upload the inference script (run from SnowSQL or Snowflake UI)
PUT file:///path/to/snowflake_inference.py @ml_models_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

### Step 2: Create the Python UDF

```sql
CREATE OR REPLACE FUNCTION predict_sentiment(text VARCHAR)
RETURNS INT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('setfit', 'sentence-transformers', 'torch', 'huggingface_hub')
IMPORTS = ('@ml_models_stage/snowflake_inference.py')
HANDLER = 'snowflake_inference.predict_sentiment';
```

### Step 3: Test the UDF

```sql
-- Test with a single value
SELECT predict_sentiment('This product is amazing!') AS sentiment;

-- Apply to a table
SELECT 
    comment_text,
    predict_sentiment(comment_text) AS predicted_sentiment
FROM customer_feedback
LIMIT 10;
```

### Step 4: Create Batch Function (Optional)

For better performance on large datasets, create a vectorized UDF:

```sql
CREATE OR REPLACE FUNCTION predict_sentiment_batch(texts ARRAY)
RETURNS ARRAY
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('setfit', 'sentence-transformers', 'torch', 'huggingface_hub')
IMPORTS = ('@ml_models_stage/snowflake_inference.py')
HANDLER = 'snowflake_inference.predict_sentiment_batch';
```

### Snowflake Configuration Notes

| Setting | Value | Notes |
|---------|-------|-------|
| Runtime | Python 3.10 | Required for SetFit compatibility |
| External Access | Enabled | Must allow `huggingface.co` for model downloads |
| Warehouse Size | Medium+ | Recommended for transformer inference |

### Enable External Network Access

The UDF needs to download the model from Hugging Face on first run:

```sql
-- Create network rule for Hugging Face
CREATE OR REPLACE NETWORK RULE hf_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('huggingface.co', 'cdn-lfs.huggingface.co');

-- Create external access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION hf_access_integration
  ALLOWED_NETWORK_RULES = (hf_network_rule)
  ENABLED = TRUE;

-- Update UDF with external access
CREATE OR REPLACE FUNCTION predict_sentiment(text VARCHAR)
RETURNS INT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('setfit', 'sentence-transformers', 'torch', 'huggingface_hub')
IMPORTS = ('@ml_models_stage/snowflake_inference.py')
HANDLER = 'snowflake_inference.predict_sentiment'
EXTERNAL_ACCESS_INTEGRATIONS = (hf_access_integration);
```

---

## Label Mapping

| Label | Meaning |
|-------|---------|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

---

## Troubleshooting

### Model not downloading in Snowflake

- Ensure external access integration is enabled
- Verify `huggingface.co` and `cdn-lfs.huggingface.co` are in the allowlist

### Out of memory errors

- Use a larger Snowflake warehouse (MEDIUM or LARGE)
- Process data in smaller batches

### Slow first prediction

- The model downloads on first call (~500MB)
- Subsequent calls use the cached model

---

## License

MIT
