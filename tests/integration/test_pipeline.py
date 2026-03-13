"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
Integration Tests: End-to-End Pipeline Validation
================================================================================
Author: Dilip Chennam
================================================================================
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import IsolationForest


# ── Full pipeline fixture
@pytest.fixture(scope='module')
def pipeline_data():
    """Generate and engineer a full dataset for integration testing."""
    np.random.seed(42)
    n = 5000

    fed_rates = {2018:1.75, 2019:2.25, 2020:0.25, 2021:0.10, 2022:3.50, 2023:5.25, 2024:5.00}
    years       = np.random.choice(list(fed_rates.keys()), n)
    loan_types  = np.random.choice(['Mortgage', 'Auto'], n, p=[0.56, 0.44])
    fed_rate    = np.array([fed_rates[y] for y in years])
    credit_score = np.clip(np.random.normal(680, 80, n).astype(int), 300, 850)
    dti_ratio    = np.clip(np.random.normal(32, 10, n), 5, 65)
    ltv_ratio    = np.clip(np.random.normal(78, 15, n), 20, 100)
    annual_income= np.clip(np.random.lognormal(11.0, 0.5, n), 20000, 500000)
    loan_amount  = np.clip(np.random.lognormal(12.0, 0.5, n), 5000, 1500000)
    interest_rate= np.clip(fed_rate + np.random.normal(3.5, 1.2, n), 2.0, 18.0)
    num_delinq   = np.random.poisson(0.3, n)
    num_inquiries= np.random.poisson(1.2, n)
    employment_yrs = np.clip(np.random.exponential(5, n), 0, 40)
    is_self_emp  = np.random.binomial(1, 0.12, n)

    default_prob = np.clip(
        0.30 * (850 - credit_score) / 550 +
        0.20 * dti_ratio / 65 +
        0.15 * ltv_ratio / 100 +
        0.10 * num_delinq / 8 +
        0.05 * (fed_rate / 6) +
        np.random.normal(0, 0.05, n),
        0.01, 0.99
    )
    is_default = (default_prob > np.random.uniform(0.3, 0.7, n)).astype(int)

    df = pd.DataFrame({
        'loan_id':           [f'LN{str(i).zfill(8)}' for i in range(n)],
        'loan_type':         loan_types,
        'origination_year':  years,
        'loan_amount':       loan_amount,
        'annual_income':     annual_income,
        'monthly_income':    annual_income / 12,
        'monthly_payment':   loan_amount * interest_rate / 100 / 12,
        'credit_score':      credit_score,
        'dti_ratio':         dti_ratio,
        'ltv_ratio':         ltv_ratio,
        'interest_rate':     interest_rate,
        'fed_funds_rate':    fed_rate,
        'num_delinquencies': num_delinq,
        'num_inquiries_12m': num_inquiries,
        'employment_years':  employment_yrs,
        'is_self_employed':  is_self_emp,
        'is_default':        is_default,
        'default_probability': default_prob,
    })

    # Feature engineering
    df['loan_to_income_ratio']    = df['loan_amount'] / df['annual_income']
    df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']
    df['rate_spread']             = df['interest_rate'] - df['fed_funds_rate']
    df['is_high_risk']            = (
        (df['credit_score'] < 620) | (df['dti_ratio'] > 43) |
        (df['ltv_ratio'] > 95)     | (df['num_delinquencies'] > 2)
    ).astype(int)
    df['risk_score'] = (
        (850 - df['credit_score']) / 550 * 40 +
        df['dti_ratio'] / 2 +
        df['ltv_ratio'] / 10 +
        df['num_delinquencies'] * 5 +
        df['num_inquiries_12m'] * 2
    )
    df['is_covid_period']     = (df['origination_year'] == 2020).astype(int)
    df['is_rate_hike_period'] = df['origination_year'].isin([2022, 2023]).astype(int)
    df['is_mortgage']         = (df['loan_type'] == 'Mortgage').astype(int)

    FEATURES = [
        'credit_score', 'dti_ratio', 'ltv_ratio', 'interest_rate',
        'loan_to_income_ratio', 'payment_to_income_ratio', 'rate_spread',
        'risk_score', 'num_delinquencies', 'num_inquiries_12m',
        'employment_years', 'is_self_employed', 'fed_funds_rate',
        'is_high_risk', 'is_mortgage', 'is_covid_period', 'is_rate_hike_period'
    ]

    X = df[FEATURES]
    y = df['is_default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return {'df': df, 'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test, 'features': FEATURES}


# ============================================================
# PIPELINE INTEGRATION TESTS
# ============================================================

class TestFullPipeline:

    def test_dataset_size(self, pipeline_data):
        """Dataset should have expected number of records."""
        assert len(pipeline_data['df']) == 5000

    def test_train_test_split_sizes(self, pipeline_data):
        """Train/test split should be 80/20."""
        total = len(pipeline_data['X_train']) + len(pipeline_data['X_test'])
        assert len(pipeline_data['X_test']) == int(total * 0.2)

    def test_no_data_leakage(self, pipeline_data):
        """Train and test sets should have no overlapping indices."""
        train_idx = set(pipeline_data['X_train'].index)
        test_idx  = set(pipeline_data['X_test'].index)
        assert len(train_idx & test_idx) == 0, "Data leakage detected"

    def test_stratified_split(self, pipeline_data):
        """Train and test default rates should be similar."""
        train_rate = pipeline_data['y_train'].mean()
        test_rate  = pipeline_data['y_test'].mean()
        assert abs(train_rate - test_rate) < 0.02, \
            f"Stratification failed: train={train_rate:.3f}, test={test_rate:.3f}"

    def test_feature_count(self, pipeline_data):
        """Feature matrix should have expected number of columns."""
        assert pipeline_data['X_train'].shape[1] == len(pipeline_data['features'])

    def test_no_nulls_in_features(self, pipeline_data):
        """Feature matrix should have no null values."""
        assert pipeline_data['X_train'].isna().sum().sum() == 0
        assert pipeline_data['X_test'].isna().sum().sum() == 0


class TestLogisticRegressionPipeline:

    def test_lr_trains_and_predicts(self, pipeline_data):
        """Logistic Regression should train and produce predictions."""
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(pipeline_data['X_train'])
        X_test_s  = scaler.transform(pipeline_data['X_test'])
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train_s, pipeline_data['y_train'])
        probs = lr.predict_proba(X_test_s)[:, 1]
        assert len(probs) == len(pipeline_data['y_test'])

    def test_lr_probabilities_valid(self, pipeline_data):
        """LR probabilities must be in [0, 1]."""
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(pipeline_data['X_train'])
        X_test_s  = scaler.transform(pipeline_data['X_test'])
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train_s, pipeline_data['y_train'])
        probs = lr.predict_proba(X_test_s)[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_lr_auc_above_random(self, pipeline_data):
        """LR AUC must exceed 0.5 (random baseline)."""
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(pipeline_data['X_train'])
        X_test_s  = scaler.transform(pipeline_data['X_test'])
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train_s, pipeline_data['y_train'])
        probs = lr.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(pipeline_data['y_test'], probs)
        assert auc > 0.5, f"LR AUC {auc:.4f} not above random baseline"


class TestXGBoostPipeline:

    def test_xgb_trains_and_predicts(self, pipeline_data):
        """XGBoost should train and produce predictions."""
        model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(pipeline_data['X_train'], pipeline_data['y_train'])
        probs = model.predict_proba(pipeline_data['X_test'])[:, 1]
        assert len(probs) == len(pipeline_data['y_test'])

    def test_xgb_auc_above_random(self, pipeline_data):
        """XGBoost AUC must exceed 0.5."""
        model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(pipeline_data['X_train'], pipeline_data['y_train'])
        probs = model.predict_proba(pipeline_data['X_test'])[:, 1]
        auc = roc_auc_score(pipeline_data['y_test'], probs)
        assert auc > 0.5, f"XGBoost AUC {auc:.4f} not above random baseline"

    def test_xgb_feature_importance_sums_to_one(self, pipeline_data):
        """XGBoost feature importances should sum to ~1."""
        model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(pipeline_data['X_train'], pipeline_data['y_train'])
        assert abs(model.feature_importances_.sum() - 1.0) < 0.01


class TestIsolationForestPipeline:

    def test_isolation_forest_flags_anomalies(self, pipeline_data):
        """Isolation Forest should flag some records as anomalies."""
        iso = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        iso.fit(pipeline_data['X_train'])
        labels = iso.predict(pipeline_data['X_test'])
        anomalies = (labels == -1).sum()
        assert anomalies > 0, "No anomalies detected"

    def test_isolation_forest_anomaly_rate(self, pipeline_data):
        """Anomaly rate should be close to contamination parameter (5%)."""
        iso = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        iso.fit(pipeline_data['X_train'])
        labels = iso.predict(pipeline_data['X_test'])
        anomaly_rate = (labels == -1).mean()
        assert 0.02 <= anomaly_rate <= 0.10, \
            f"Anomaly rate {anomaly_rate:.2%} outside expected range"
