# Operations Runbook

## 1) Data download

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/home-credit-default-risk
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-fraud-detection
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/mlg-ulb_creditcardfraud
```

## 2) Train models

```bash
python mlops/train_pipeline.py --task fraud --data data/processed/fraud_training.csv
python mlops/train_pipeline.py --task credit --data data/processed/credit_training.csv
```

## 3) Run API

```bash
uvicorn api.service:app --host 0.0.0.0 --port 8000
```

## 4) Drift monitoring

```bash
python mlops/drift_monitor.py --reference data/processed/fraud_reference.csv --production data/processed/fraud_production.csv --column amount
```
