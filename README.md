# Unified Financial Risk Intelligence System (Fraud + Credit Risk)

Production-grade ML platform for Indian banks / NBFCs covering fraud detection, credit risk scoring, early warning signals, explainability, and API serving.

## Why this project is interview-ready (20–25 LPA role standard)

- End-to-end modular architecture (data -> features -> models -> explainability -> API -> MLOps).
- Real public datasets (Kaggle competitions/datasets) instead of toy synthetic-only samples.
- Business-first evaluation: ROC-AUC + PR-AUC + KS + confusion + cost-weighted thresholding.
- RBI-style governance primitives: model traceability, explainability summary, drift checks.

---

## 1) Business context and Indian NBFC mapping

### Objectives
1. **Transaction Fraud Detection** (real-time scoring + batch retraining).
2. **Customer Credit Risk Scoring** for underwriting and limit/EMI decisions.
3. **Early Warning Signals (EWS)** via delinquency and drift trend monitoring.
4. **Explainable AI** to support compliance and analyst investigation.

### Dataset choices (real and public)

| Problem | Dataset | Why it maps to Indian BFSI / NBFC reality |
|---|---|---|
| Credit risk | Home Credit Default Risk | Similar retail loan underwriting signals: bureau behavior, demographics, prior installment history. |
| Credit risk | Give Me Some Credit | Delinquency and utilization patterns align to unsecured lending and EMI risk segmentation. |
| Fraud | IEEE-CIS Fraud Detection | Device + identity + transaction behavior suitable for UPI/card-not-present fraud controls. |
| Fraud | Credit Card Fraud (ULB) | Real imbalanced fraud benchmark useful for rare-event learning and threshold design. |

> This repo includes ingestion scripts for pulling these sources via Kaggle API and shaping them into bank-ready features.

---

## 2) Repository structure

```text
/data
  /raw
  /processed
  /feature_store
/src
  /data_ingestion
  /data_validation
  /feature_engineering
  /model_training
  /model_evaluation
  /explainability
  /drift_detection
  /utils
/api
/mlops
/notebooks
/tests
/configs
/docs
```

---

## 3) Feature engineering highlights

### Fraud feature patterns
- Transaction velocity windows: `txn_count_1h`, `txn_count_24h`, `txn_amount_7d`.
- Geo anomaly signal: `geo_distance_km` using haversine distance.
- Device/session behavior placeholder in API schema (`device_change_flag`).
- Merchant risk proxy support (`merchant_risk_score`) for high-risk category tracking.

### Credit risk feature patterns
- `income_to_emi_ratio` for affordability stress.
- `utilization_ratio` for revolving credit risk.
- `delinquency_trend_90d` from recent DPD trajectory.
- Bureau and tenure related fields in serving schema.

---

## 4) Modeling approach

Implemented and comparable model families:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (advanced gradient boosting)

Training pipeline:
- Handles numerical + categorical columns via unified sklearn `ColumnTransformer`.
- Class imbalance strategy via class weights / boosted tree handling.
- MLflow tracking for each run + metric log.
- Best model selected by validation ROC-AUC and persisted as a reusable pipeline artifact.

---

## 5) Evaluation framework (business-driven)

`src/model_evaluation/evaluate.py` includes:
- ROC-AUC
- PR-AUC (critical for fraud imbalance)
- KS statistic
- Confusion matrix
- Threshold optimization using business cost: `FN_COST >> FP_COST`

---

## 6) Explainability and compliance

- Feature ranking helper for global importance (tree importances / linear coefficients).
- Local explanation endpoint support (`local_explanation`) with SHAP-first and safe fallback.
- API returns **score + threshold + decision + explanation summary**, suitable for analyst UI and audit logging.

---

## 7) FastAPI serving layer

Endpoints:
- `GET /health`
- `GET /model-info`
- `POST /predict/fraud`
- `POST /predict/credit-risk`

All requests validated through Pydantic schemas; responses include confidence score and decision explanation.

---

## 8) MLOps readiness

- MLflow experiment tracking configured in `configs/training.yaml`.
- Model artifacts versioned under `artifacts/models`.
- Dockerized API deployment (`Dockerfile`).
- Drift detection utility via PSI (`mlops/drift_monitor.py`).

---

## 9) Quickstart

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download datasets (real public sources)
```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/home-credit-default-risk
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-fraud-detection
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/mlg-ulb_creditcardfraud
```

### Train
```bash
python mlops/train_pipeline.py --task fraud --data data/processed/fraud_training.csv
python mlops/train_pipeline.py --task credit --data data/processed/credit_training.csv
```

### Serve
```bash
uvicorn api.service:app --host 0.0.0.0 --port 8000
```

---

## 10) Architecture diagram

See: [`docs/architecture.md`](docs/architecture.md)

---

## 11) Interview talking points

- How to calibrate thresholds by portfolio segment (rural/urban, product type).
- How to convert model output into action policy (step-up auth / manual review / reject).
- How to manage model risk: explainability, drift triggers, retraining SLAs, rollback plan.
- How to align fraud and credit pipelines into a single risk intelligence fabric.
