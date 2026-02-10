# System Architecture

```text
                    +-----------------------------+
                    |  Kaggle Datasets (Real)    |
                    |  - Home Credit             |
                    |  - Give Me Some Credit     |
                    |  - IEEE-CIS Fraud          |
                    +-------------+---------------+
                                  |
                                  v
+------------------+    +-----------------------+    +------------------------+
| data_ingestion   | -> | data_validation       | -> | feature_engineering    |
| Kaggle API pull  |    | schema/null/outliers  |    | velocity, geo, credit  |
+------------------+    +-----------------------+    +-----------+------------+
                                                               |
                                                               v
                                      +------------------------+-----------------------+
                                      | model_training + mlflow tracking               |
                                      | LR / RF / XGBoost + class balancing + tuning   |
                                      +------------------------+-----------------------+
                                                               |
                                                               v
                                      +------------------------+-----------------------+
                                      | model_evaluation + threshold optimization       |
                                      | ROC-AUC, PR-AUC, KS, confusion, business cost  |
                                      +------------------------+-----------------------+
                                                               |
                         +-------------------------------------+---------------------------------+
                         |                                                                       |
                         v                                                                       v
         +-------------------------------+                                         +----------------------------+
         | explainability                |                                         | drift_detection            |
         | SHAP/local reason summary     |                                         | PSI + retraining trigger   |
         +-------------------------------+                                         +----------------------------+
                         |
                         v
                +------------------+
                | FastAPI Serving  |
                | /predict/fraud   |
                | /predict/credit  |
                +------------------+
```

## RBI-style model governance alignment

- Traceability through `mlflow` run IDs and artifact versioning.
- Explainability layer for adverse action and fraud decision rationale.
- Drift monitoring using PSI thresholds to trigger controlled retraining.
- Clear score + threshold + reason output for audit trails.
