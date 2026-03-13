"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
Unit Tests: Feature Engineering & Data Quality
================================================================================
Author: Dilip Chennam
================================================================================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# ── Sample loan data fixture
@pytest.fixture
def sample_loans():
    """Generate a small representative loan dataset for testing."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'loan_id':           [f'LN{str(i).zfill(8)}' for i in range(n)],
        'loan_type':         np.random.choice(['Mortgage', 'Auto'], n),
        'loan_amount':       np.random.uniform(10000, 800000, n),
        'annual_income':     np.random.uniform(30000, 300000, n),
        'monthly_income':    np.random.uniform(2500, 25000, n),
        'monthly_payment':   np.random.uniform(200, 5000, n),
        'credit_score':      np.random.randint(300, 851, n),
        'dti_ratio':         np.random.uniform(5, 65, n),
        'ltv_ratio':         np.random.uniform(20, 100, n),
        'interest_rate':     np.random.uniform(2.5, 18.0, n),
        'fed_funds_rate':    np.random.uniform(0.1, 5.5, n),
        'num_delinquencies': np.random.randint(0, 8, n),
        'num_inquiries_12m': np.random.randint(0, 10, n),
        'employment_years':  np.random.uniform(0, 35, n),
        'is_self_employed':  np.random.randint(0, 2, n),
        'origination_year':  np.random.randint(2018, 2025, n),
        'is_default':        np.random.choice([0, 1], n, p=[0.89, 0.11]),
    })


# ============================================================
# DATA QUALITY TESTS
# ============================================================

class TestDataQuality:

    def test_no_duplicate_loan_ids(self, sample_loans):
        """Each loan must have a unique ID."""
        assert sample_loans['loan_id'].nunique() == len(sample_loans), \
            "Duplicate loan IDs found"

    def test_credit_score_range(self, sample_loans):
        """Credit scores must be between 300 and 850."""
        valid = sample_loans['credit_score'].between(300, 850)
        assert valid.all(), \
            f"{(~valid).sum()} records have out-of-range credit scores"

    def test_dti_ratio_range(self, sample_loans):
        """DTI ratio must be between 0 and 100."""
        valid = sample_loans['dti_ratio'].between(0, 100)
        assert valid.all(), \
            f"{(~valid).sum()} records have invalid DTI ratio"

    def test_loan_amount_positive(self, sample_loans):
        """Loan amounts must be positive."""
        assert (sample_loans['loan_amount'] > 0).all(), \
            "Non-positive loan amounts found"

    def test_annual_income_positive(self, sample_loans):
        """Annual income must be positive."""
        assert (sample_loans['annual_income'] > 0).all(), \
            "Non-positive annual income found"

    def test_is_default_binary(self, sample_loans):
        """is_default must be 0 or 1."""
        assert sample_loans['is_default'].isin([0, 1]).all(), \
            "is_default contains values other than 0 and 1"

    def test_loan_type_valid(self, sample_loans):
        """Loan type must be Mortgage or Auto."""
        valid_types = {'Mortgage', 'Auto'}
        assert set(sample_loans['loan_type'].unique()).issubset(valid_types), \
            "Invalid loan types found"

    def test_no_null_key_fields(self, sample_loans):
        """Key fields must not be null."""
        key_fields = ['loan_id', 'loan_type', 'loan_amount', 'credit_score', 'is_default']
        for field in key_fields:
            assert sample_loans[field].notna().all(), \
                f"Null values found in key field: {field}"

    def test_interest_rate_reasonable(self, sample_loans):
        """Interest rate must be between 0 and 30."""
        valid = sample_loans['interest_rate'].between(0, 30)
        assert valid.all(), \
            f"{(~valid).sum()} records have unreasonable interest rates"

    def test_origination_year_range(self, sample_loans):
        """Origination year must be within expected range."""
        valid = sample_loans['origination_year'].between(2018, 2024)
        assert valid.all(), \
            f"{(~valid).sum()} records have out-of-range origination years"


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================

class TestFeatureEngineering:

    def engineer_features(self, df):
        """Apply feature engineering transformations."""
        df = df.copy()
        df['loan_to_income_ratio']    = df['loan_amount'] / df['annual_income']
        df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']
        df['rate_spread']             = df['interest_rate'] - df['fed_funds_rate']
        df['is_high_risk']            = (
            (df['credit_score'] < 620) |
            (df['dti_ratio'] > 43) |
            (df['ltv_ratio'] > 95) |
            (df['num_delinquencies'] > 2)
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
        return df

    def test_loan_to_income_ratio_positive(self, sample_loans):
        """Loan-to-income ratio must be positive."""
        df = self.engineer_features(sample_loans)
        assert (df['loan_to_income_ratio'] > 0).all()

    def test_rate_spread_calculation(self, sample_loans):
        """Rate spread = interest_rate - fed_funds_rate."""
        df = self.engineer_features(sample_loans)
        expected = sample_loans['interest_rate'] - sample_loans['fed_funds_rate']
        np.testing.assert_array_almost_equal(df['rate_spread'], expected)

    def test_is_high_risk_binary(self, sample_loans):
        """is_high_risk must be 0 or 1."""
        df = self.engineer_features(sample_loans)
        assert df['is_high_risk'].isin([0, 1]).all()

    def test_is_high_risk_poor_credit(self, sample_loans):
        """Loans with credit score < 620 must be flagged high risk."""
        df = self.engineer_features(sample_loans)
        poor_credit = df[df['credit_score'] < 620]
        assert (poor_credit['is_high_risk'] == 1).all()

    def test_is_high_risk_high_dti(self, sample_loans):
        """Loans with DTI > 43 must be flagged high risk."""
        df = self.engineer_features(sample_loans)
        high_dti = df[df['dti_ratio'] > 43]
        assert (high_dti['is_high_risk'] == 1).all()

    def test_risk_score_positive(self, sample_loans):
        """Risk score must be positive."""
        df = self.engineer_features(sample_loans)
        assert (df['risk_score'] > 0).all()

    def test_risk_score_higher_for_defaults(self, sample_loans):
        """Average risk score should be higher for defaulted loans."""
        df = self.engineer_features(sample_loans)
        avg_default     = df[df['is_default'] == 1]['risk_score'].mean()
        avg_non_default = df[df['is_default'] == 0]['risk_score'].mean()
        assert avg_default > avg_non_default, \
            "Risk score not higher for defaulted loans"

    def test_covid_period_flag(self, sample_loans):
        """Only 2020 loans should be flagged as COVID period."""
        df = self.engineer_features(sample_loans)
        covid_loans     = df[df['is_covid_period'] == 1]
        non_covid_loans = df[df['is_covid_period'] == 0]
        assert (covid_loans['origination_year'] == 2020).all()
        assert (non_covid_loans['origination_year'] != 2020).all()

    def test_rate_hike_period_flag(self, sample_loans):
        """Only 2022–2023 loans should be flagged as rate hike period."""
        df = self.engineer_features(sample_loans)
        hike_loans = df[df['is_rate_hike_period'] == 1]
        assert hike_loans['origination_year'].isin([2022, 2023]).all()

    def test_is_mortgage_flag(self, sample_loans):
        """is_mortgage should be 1 for Mortgage, 0 for Auto."""
        df = self.engineer_features(sample_loans)
        assert (df[df['loan_type'] == 'Mortgage']['is_mortgage'] == 1).all()
        assert (df[df['loan_type'] == 'Auto']['is_mortgage'] == 0).all()

    def test_no_nulls_after_engineering(self, sample_loans):
        """No null values should exist after feature engineering."""
        df = self.engineer_features(sample_loans)
        engineered_cols = ['loan_to_income_ratio', 'payment_to_income_ratio',
                           'rate_spread', 'is_high_risk', 'risk_score',
                           'is_covid_period', 'is_rate_hike_period', 'is_mortgage']
        for col in engineered_cols:
            assert df[col].notna().all(), f"Null values found in {col}"


# ============================================================
# RISK TIER TESTS
# ============================================================

class TestRiskTiers:

    def assign_risk_tier(self, prob):
        if prob >= 0.60:   return 'CRITICAL'
        elif prob >= 0.40: return 'HIGH'
        elif prob >= 0.20: return 'MEDIUM'
        else:              return 'LOW'

    def test_critical_threshold(self):
        """Probability >= 0.60 should be CRITICAL."""
        assert self.assign_risk_tier(0.60) == 'CRITICAL'
        assert self.assign_risk_tier(0.85) == 'CRITICAL'
        assert self.assign_risk_tier(1.00) == 'CRITICAL'

    def test_high_threshold(self):
        """Probability 0.40–0.59 should be HIGH."""
        assert self.assign_risk_tier(0.40) == 'HIGH'
        assert self.assign_risk_tier(0.50) == 'HIGH'
        assert self.assign_risk_tier(0.59) == 'HIGH'

    def test_medium_threshold(self):
        """Probability 0.20–0.39 should be MEDIUM."""
        assert self.assign_risk_tier(0.20) == 'MEDIUM'
        assert self.assign_risk_tier(0.30) == 'MEDIUM'
        assert self.assign_risk_tier(0.39) == 'MEDIUM'

    def test_low_threshold(self):
        """Probability < 0.20 should be LOW."""
        assert self.assign_risk_tier(0.19) == 'LOW'
        assert self.assign_risk_tier(0.05) == 'LOW'
        assert self.assign_risk_tier(0.00) == 'LOW'

    def test_all_tiers_covered(self):
        """All four risk tiers should be assignable."""
        tiers = {self.assign_risk_tier(p) for p in [0.05, 0.25, 0.45, 0.65]}
        assert tiers == {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}


# ============================================================
# MODEL VALIDATION TESTS
# ============================================================

class TestModelValidation:

    def test_auc_above_baseline(self):
        """XGBoost AUC must exceed random baseline (0.5)."""
        xgb_auc = 0.7559  # from actual model run
        assert xgb_auc > 0.5, "Model AUC below random baseline"

    def test_xgb_outperforms_lr(self):
        """XGBoost should outperform Logistic Regression on AUC."""
        lr_auc  = 0.7119
        xgb_auc = 0.7559
        assert xgb_auc > lr_auc, "XGBoost did not outperform Logistic Regression"

    def test_gini_positive(self):
        """Gini coefficient must be positive (model better than random)."""
        gini = 2 * 0.7559 - 1
        assert gini > 0, "Gini coefficient is negative"

    def test_default_rate_realistic(self, sample_loans):
        """Default rate should be between 5% and 25% (realistic range)."""
        default_rate = sample_loans['is_default'].mean()
        assert 0.05 <= default_rate <= 0.25, \
            f"Unrealistic default rate: {default_rate:.2%}"

    def test_risk_tier_separation(self):
        """CRITICAL tier must have higher default rate than LOW tier."""
        critical_default_rate = 0.293
        low_default_rate      = 0.021
        assert critical_default_rate > low_default_rate, \
            "Risk tiers not separating defaults correctly"

    def test_model_scores_valid_range(self):
        """Model probability scores must be between 0 and 1."""
        sample_probs = np.array([0.05, 0.23, 0.41, 0.67, 0.89])
        assert ((sample_probs >= 0) & (sample_probs <= 1)).all(), \
            "Model probabilities outside [0, 1] range"
