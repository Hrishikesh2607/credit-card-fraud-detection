# Credit Card Fraud Detection

A production-grade ML system that detects fraudulent credit card
transactions in real time. Trained on 284,807 transactions with
an extreme class imbalance (0.17% fraud).

## Results

| Metric     | Value  |
|------------|--------|
| AUPRC      | 0.88   |
| ROC-AUC    | 0.982  |
| Precision  | 90.9%  |
| Recall     | 81.6%  |
| F1 (fraud) | 0.845  |
| Threshold  | ~0.15  |

> AUPRC is the primary metric — ROC-AUC is misleading for
> datasets with 1:578 class imbalance.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/yourname/fraud-detection
cd fraud-detection
python -m venv fraud-env && source fraud-env/bin/activate
pip install -r requirements.txt

# 2. Download data
kaggle datasets download mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/

# 3. Run full pipeline
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_features.ipynb
jupyter notebook notebooks/03_models.ipynb

# 4. Start the API
cd api/ && uvicorn main:app --reload

# 5. Or run with Docker
docker-compose up
```

## API usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.36, ..., "Amount": 149.62, "Time": 0.0}'

# Response
{
  "fraud_probability": 0.0231,
  "flagged": false,
  "risk_tier": "low",
  "threshold_used": 0.30
}
```

## Project structure

```
fraud-detection/
├── data/               # raw CSV (gitignored) + parquet splits
├── notebooks/          # 01_eda, 02_features, 03_models
├── models/             # best_model.pkl, scaler.pkl, threshold_config.json
├── api/                # FastAPI service
│   ├── main.py
│   ├── schemas.py
│   ├── preprocess.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Key design decisions

- **XGBoost over RandomForest**: scale_pos_weight=578 handles
  imbalance natively; no SMOTE needed for gradient boosting
- **AUPRC not accuracy**: 99.83% accuracy is achievable by
  predicting "legit" for everything — completely useless
- **Cost-optimized threshold**: tuned to minimise
  FN×$150 + FP×$5, not F1 alone
- **Training-serving parity**: preprocess.py mirrors
  02_features.ipynb exactly to prevent serving skew

## 👤 Author

**Hrishikesh Ganji**
