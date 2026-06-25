# Credit Risk Early Warning System
### Meridian Bank | End-to-End ML Pipeline for Default Prediction & Risk Stratification

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Glue%20%7C%20Lambda-orange?logo=amazonaws)
![Databricks](https://img.shields.io/badge/Databricks-ML%20Platform-red?logo=databricks)
![Snowflake](https://img.shields.io/badge/Snowflake-Data%20Warehouse-29B5E8?logo=snowflake)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?logo=tableau)
![Tests](https://img.shields.io/badge/Tests-46%20Passing-brightgreen)
![XGBoost AUC](https://img.shields.io/badge/XGBoost%20AUC-0.7559-success)

---

## Overview

Built a production-grade credit risk pipeline for **Meridian Bank** (fictional) to identify high-risk borrowers before loan origination. The system ingests 250,000 mortgage and auto loans spanning **2018–2024** — covering the full economic cycle including the COVID shock and Fed rate hike cycle — and applies machine learning to predict defaults with 14x risk tier separation.

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 250,000 loans |
| Default rate | 11.03% |
| XGBoost AUC-ROC | 0.7559 |
| XGBoost Gini | 0.5118 |
| KS Statistic | 0.3954 |
| CRITICAL tier default rate | 29.3% |
| LOW tier default rate | 2.1% |
| **Risk tier separation** | **14x** |
| Test coverage | 46 tests passing |

---

## Architecture

```
Raw CSVs (S3)
     │
     ▼
AWS Lambda ──► triggers on S3 PUT
     │
     ▼
AWS Glue ETL ──► DQ checks, feature engineering, Parquet output
     │
     ▼
Snowflake ──► RAW → ANALYTICS → ML → REPORTING
     │              │
     │         8 analytical views
     │         Stored procedures
     │
     ▼
Databricks ──► Feature engineering, SMOTE balancing
     │          Logistic Regression | XGBoost | Isolation Forest
     │          SHAP explainability
     │
     ▼
Tableau ──► 3-page executive dashboard
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud Infrastructure | AWS S3, AWS Glue, AWS Lambda |
| ML Platform | Databricks (PySpark, Python notebooks) |
| Data Warehouse | Snowflake (4 schemas, 8 views, stored procedures) |
| ML Models | XGBoost, Logistic Regression, Isolation Forest, SHAP |
| Visualization | Tableau Desktop |
| Languages | Python, SQL, PySpark |
| Testing | pytest (46 tests) |
| DevOps | Git, GitHub |

---

## Repository Structure

```
credit-risk-early-warning/
├── data/
│   ├── generate_loan_data.py       # Synthetic dataset generator (250K loans)
│   └── schemas/
│       └── data_dictionary.md      # Full field definitions
├── aws/
│   ├── s3_configs/
│   │   └── setup_s3.py             # S3 bucket provisioning + data upload
│   ├── glue/
│   │   ├── glue_etl_job.py         # PySpark ETL: DQ + feature engineering + Parquet
│   │   └── provision_glue_job.py   # Glue job registration via boto3
│   └── lambda/
│       ├── s3_trigger.py           # Lambda: auto-triggers Glue on S3 file arrival
│       └── deploy_lambda.py        # Lambda deployment script
├── snowflake/
│   ├── ddl/
│   │   ├── 01_setup.sql            # Database, schemas, tables
│   │   └── 02_load_data.sql        # COPY INTO + RAW → ANALYTICS transform
│   ├── views/
│   │   └── 03_analytical_views.sql # 8 reporting views
│   └── procedures/
│       └── 04_stored_procedures.sql # Model score loading, risk tier refresh
├── databricks/
│   └── notebooks/
│       └── 01_feature_engineering_and_ml.ipynb  # Full ML notebook
├── tableau/
│   └── meridian_credit_risk.twbx   # Packaged Tableau workbook
├── tests/
│   ├── unit/
│   │   └── test_feature_engineering.py  # 32 unit tests
│   └── integration/
│       └── test_pipeline.py             # 14 integration tests
└── README.md
```

---

## Dataset

- **250,000 loans** — Mortgage (56%) and Auto (44%)
- **Date range:** 2018–2024 (full economic cycle)
- **40+ features:** borrower profile, loan terms, credit metrics, economic context
- **Default rate:** 11.03% (realistic class imbalance)
- **Economic regimes embedded:** COVID shock (2020), Fed rate hike cycle (2022–2023)

Key fields: `credit_score`, `dti_ratio`, `ltv_ratio`, `loan_amount`, `annual_income`, `interest_rate`, `fed_funds_rate`, `num_delinquencies`, `is_default`

---

## ML Pipeline (Databricks)

### Feature Engineering
- `risk_score` — composite score: credit, DTI, LTV, delinquencies, inquiries
- `loan_to_income_ratio`, `payment_to_income_ratio`, `rate_spread`
- `is_high_risk` — flag: credit < 620 OR DTI > 43 OR LTV > 95 OR delinq > 2
- `is_covid_period`, `is_rate_hike_period` — economic regime flags
- Credit score, DTI, income buckets (ordinal encoded)

### Class Imbalance
SMOTE applied to training set: 11.3% → 50% balanced for model training.

### Model Results

| Model | AUC-ROC | Gini | KS Stat | Role |
|-------|---------|------|---------|------|
| Logistic Regression | 0.7119 | 0.4238 | 0.3144 | Baseline |
| XGBoost | 0.7559 | 0.5118 | 0.3954 | Primary |
| Ensemble (LR + XGB) | 0.7505 | 0.5011 | — | Production |

### SHAP Feature Importance (XGBoost)
1. `risk_score` — ↑ increases default risk
2. `credit_score` — ↓ decreases default risk
3. `ltv_ratio` — ↑ increases default risk
4. `is_self_employed` — ↑ increases default risk
5. `dti_ratio` — ↑ increases default risk

### Anomaly Detection
Isolation Forest flags 4.8% of loans as anomalous, with 15.7% overlap with actual defaults.

---

## Risk Tier Framework

| Tier | Probability Threshold | Default Rate | Count |
|------|----------------------|--------------|-------|
| CRITICAL | ≥ 0.60 | 29.3% | 1,034 |
| HIGH | 0.40 – 0.59 | 19.0% | 2,426 |
| MEDIUM | 0.20 – 0.39 | 8.9% | 3,423 |
| LOW | < 0.20 | 2.1% | 3,117 |

**14x separation** between CRITICAL and LOW tiers.

---

## Snowflake Schema

```sql
MERIDIAN_CREDIT_RISK
├── RAW.LOANS_RAW              -- 250K raw loan records
├── ANALYTICS.LOANS_CLEAN     -- Cleaned + 15 engineered features
├── ML.MODEL_SCORES           -- LR, XGBoost, Ensemble, Isolation Forest scores
├── ML.MODEL_PERFORMANCE      -- AUC, Gini, KS tracked across model versions
└── REPORTING (8 views)
    ├── VW_PORTFOLIO_OVERVIEW
    ├── VW_DEFAULT_BY_CREDIT_TIER
    ├── VW_GEOGRAPHIC_RISK
    ├── VW_ECONOMIC_REGIME
    ├── VW_RISK_SEGMENTATION
    ├── VW_EARLY_WARNING
    ├── VW_MODEL_PERFORMANCE
    └── VW_SECTOR_RISK
```

---

## Tableau Dashboard

3-page executive dashboard:
- **Page 1:** Default rate trend 2018–2024 with loan volume (combo chart) — COVID spike clearly visible
- **Page 2:** Default rate by credit score tier (Mortgage vs Auto)
- **Page 3:** Geographic risk heatmap — state-level default concentration

---

## Running the Tests

```bash
pip install pytest pandas numpy scikit-learn xgboost
pytest tests/ -v
```

Expected output: **46 passed**

---

## Running the Pipeline

```bash
# 1. Generate dataset
python data/generate_loan_data.py

# 2. Provision AWS infrastructure
python aws/s3_configs/setup_s3.py
python aws/glue/provision_glue_job.py
python aws/lambda/deploy_lambda.py

# 3. Run Snowflake DDL (in Snowflake worksheet)
# Execute: snowflake/ddl/01_setup.sql
# Execute: snowflake/ddl/02_load_data.sql
# Execute: snowflake/views/03_analytical_views.sql
# Execute: snowflake/procedures/04_stored_procedures.sql

# 4. Run Databricks notebook
# Upload: databricks/notebooks/01_feature_engineering_and_ml.ipynb
```

---

## Author

**Dilip Chennam** | [GitHub](https://github.com/Dilipchennam3005)
