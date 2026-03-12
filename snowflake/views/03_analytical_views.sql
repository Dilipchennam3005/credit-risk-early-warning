-- ============================================================
-- Meridian Bank — Credit Risk Early Warning System
-- Snowflake: Analytical Views
-- File: snowflake/views/03_analytical_views.sql
-- Author: Dilip Chennam
-- ============================================================

USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
USE DATABASE MERIDIAN_CREDIT_RISK;
USE SCHEMA REPORTING;

-- ============================================================
-- VIEW 1: Portfolio Overview (feeds Tableau KPI page)
-- ============================================================
CREATE OR REPLACE VIEW VW_PORTFOLIO_OVERVIEW AS
SELECT
    LOAN_TYPE,
    ORIGINATION_YEAR,
    COUNT(*)                                        AS total_loans,
    SUM(LOAN_AMOUNT)                                AS total_exposure,
    ROUND(AVG(LOAN_AMOUNT), 2)                      AS avg_loan_amount,
    SUM(IS_DEFAULT)                                 AS total_defaults,
    ROUND(AVG(IS_DEFAULT) * 100, 2)                 AS default_rate_pct,
    ROUND(AVG(CREDIT_SCORE), 0)                     AS avg_credit_score,
    ROUND(AVG(DTI_RATIO), 1)                        AS avg_dti,
    ROUND(AVG(INTEREST_RATE), 2)                    AS avg_interest_rate,
    ROUND(AVG(LTV_RATIO), 1)                        AS avg_ltv,
    SUM(IS_HIGH_RISK)                               AS high_risk_loans,
    ROUND(AVG(RISK_SCORE), 1)                       AS avg_risk_score,
    ROUND(SUM(IS_HIGH_RISK) / COUNT(*) * 100, 2)    AS high_risk_pct,
    -- Dollar exposure at risk (defaults * avg loan amount)
    ROUND(SUM(CASE WHEN IS_DEFAULT = 1 THEN LOAN_AMOUNT ELSE 0 END), 2) AS exposure_at_risk
FROM ANALYTICS.LOANS_CLEAN
GROUP BY LOAN_TYPE, ORIGINATION_YEAR
ORDER BY ORIGINATION_YEAR, LOAN_TYPE;

-- ============================================================
-- VIEW 2: Default Rate by Credit Tier and Year (trend analysis)
-- ============================================================
CREATE OR REPLACE VIEW VW_DEFAULT_BY_CREDIT_TIER AS
SELECT
    ORIGINATION_YEAR,
    CREDIT_TIER,
    CREDIT_SCORE_BUCKET,
    LOAN_TYPE,
    COUNT(*)                                        AS loan_count,
    SUM(IS_DEFAULT)                                 AS defaults,
    ROUND(AVG(IS_DEFAULT) * 100, 2)                 AS default_rate_pct,
    ROUND(AVG(CREDIT_SCORE), 0)                     AS avg_credit_score,
    ROUND(AVG(LOAN_AMOUNT), 2)                      AS avg_loan_amount,
    ROUND(SUM(LOAN_AMOUNT), 2)                      AS total_exposure,
    ROUND(SUM(CASE WHEN IS_DEFAULT = 1 THEN LOAN_AMOUNT ELSE 0 END), 2) AS defaulted_exposure
FROM ANALYTICS.LOANS_CLEAN
GROUP BY ORIGINATION_YEAR, CREDIT_TIER, CREDIT_SCORE_BUCKET, LOAN_TYPE
ORDER BY ORIGINATION_YEAR, CREDIT_SCORE_BUCKET;

-- ============================================================
-- VIEW 3: Risk Segmentation Matrix (DTI vs LTV vs Default)
-- ============================================================
CREATE OR REPLACE VIEW VW_RISK_SEGMENTATION AS
SELECT
    DTI_BUCKET,
    LTV_BUCKET,
    LOAN_TYPE,
    COUNT(*)                                        AS loan_count,
    ROUND(AVG(IS_DEFAULT) * 100, 2)                 AS default_rate_pct,
    ROUND(AVG(CREDIT_SCORE), 0)                     AS avg_credit_score,
    ROUND(AVG(RISK_SCORE), 1)                       AS avg_risk_score,
    ROUND(AVG(INTEREST_RATE), 2)                    AS avg_interest_rate,
    ROUND(SUM(LOAN_AMOUNT) / 1000000, 2)            AS total_exposure_millions
FROM ANALYTICS.LOANS_CLEAN
GROUP BY DTI_BUCKET, LTV_BUCKET, LOAN_TYPE
ORDER BY default_rate_pct DESC;

-- ============================================================
-- VIEW 4: Geographic Risk Analysis (state-level heatmap)
-- ============================================================
CREATE OR REPLACE VIEW VW_GEOGRAPHIC_RISK AS
SELECT
    STATE,
    LOAN_TYPE,
    COUNT(*)                                        AS loan_count,
    ROUND(AVG(IS_DEFAULT) * 100, 2)                 AS default_rate_pct,
    ROUND(SUM(LOAN_AMOUNT) / 1000000, 2)            AS total_exposure_millions,
    ROUND(AVG(CREDIT_SCORE), 0)                     AS avg_credit_score,
    ROUND(AVG(DTI_RATIO), 1)                        AS avg_dti,
    ROUND(AVG(RISK_SCORE), 1)                       AS avg_risk_score,
    SUM(IS_DEFAULT)                                 AS total_defaults,
    ROUND(SUM(CASE WHEN IS_DEFAULT=1 THEN LOAN_AMOUNT ELSE 0 END)/1000000, 2) AS defaulted_exposure_millions
FROM ANALYTICS.LOANS_CLEAN
GROUP BY STATE, LOAN_TYPE
ORDER BY default_rate_pct DESC;

-- ============================================================
-- VIEW 5: Economic Regime Analysis (COVID + Rate Hike impact)
-- ============================================================
CREATE OR REPLACE VIEW VW_ECONOMIC_REGIME AS
SELECT
    ORIGINATION_YEAR,
    ORIGINATION_QUARTER,
    LOAN_TYPE,
    ROUND(AVG(FED_FUNDS_RATE), 2)                   AS avg_fed_rate,
    COUNT(*)                                        AS originations,
    ROUND(SUM(LOAN_AMOUNT)/1000000, 2)              AS volume_millions,
    ROUND(AVG(IS_DEFAULT) * 100, 2)                 AS default_rate_pct,
    ROUND(AVG(INTEREST_RATE), 2)                    AS avg_rate,
    ROUND(AVG(CREDIT_SCORE), 0)                     AS avg_credit_score,
    MAX(IS_COVID_PERIOD)                            AS is_covid_quarter,
    MAX(IS_RATE_HIKE_PERIOD)                        AS is_rate_hike_quarter,
    -- QoQ change in default rate
    ROUND(AVG(IS_DEFAULT) * 100 -
        LAG(AVG(IS_DEFAULT) * 100) OVER (
            PARTITION BY LOAN_TYPE
            ORDER BY ORIGINATION_YEAR, ORIGINATION_QUARTER
        ), 2)                                       AS default_rate_qoq_change
FROM ANALYTICS.LOANS_CLEAN
GROUP BY ORIGINATION_YEAR, ORIGINATION_QUARTER, LOAN_TYPE
ORDER BY ORIGINATION_YEAR, ORIGINATION_QUARTER, LOAN_TYPE;

-- ============================================================
-- VIEW 6: Early Warning Signals (high-risk pipeline)
-- ============================================================
CREATE OR REPLACE VIEW VW_EARLY_WARNING AS
WITH risk_ranked AS (
    SELECT
        l.*,
        s.XGB_DEFAULT_PROB,
        s.ENSEMBLE_DEFAULT_PROB,
        s.RISK_TIER,
        s.IF_IS_ANOMALY,
        NTILE(10) OVER (ORDER BY l.RISK_SCORE DESC)     AS RISK_DECILE
    FROM ANALYTICS.LOANS_CLEAN l
    LEFT JOIN ML.MODEL_SCORES s
        ON l.LOAN_ID = s.LOAN_ID
        AND s.MODEL_VERSION = 'v1.0'
)
SELECT
    LOAN_ID,
    LOAN_TYPE,
    ORIGINATION_DATE,
    LOAN_AMOUNT,
    CREDIT_SCORE,
    CREDIT_TIER,
    DTI_RATIO,
    LTV_RATIO,
    RISK_SCORE,
    RISK_DECILE,
    RISK_TIER,
    XGB_DEFAULT_PROB,
    ENSEMBLE_DEFAULT_PROB,
    IF_IS_ANOMALY,
    LOAN_STATUS,
    IS_DEFAULT,
    -- Warning flags
    CASE WHEN CREDIT_SCORE < 580           THEN 1 ELSE 0 END AS FLAG_POOR_CREDIT,
    CASE WHEN DTI_RATIO > 43               THEN 1 ELSE 0 END AS FLAG_HIGH_DTI,
    CASE WHEN LTV_RATIO > 95               THEN 1 ELSE 0 END AS FLAG_HIGH_LTV,
    CASE WHEN NUM_DELINQUENCIES > 2        THEN 1 ELSE 0 END AS FLAG_DELINQUENCY,
    CASE WHEN NUM_INQUIRIES_12M > 5        THEN 1 ELSE 0 END AS FLAG_INQUIRY_SPIKE,
    CASE WHEN ENSEMBLE_DEFAULT_PROB > 0.50 THEN 1 ELSE 0 END AS FLAG_MODEL_HIGH_RISK,
    CASE WHEN IF_IS_ANOMALY = 1            THEN 1 ELSE 0 END AS FLAG_ANOMALY,
    -- Total warning flags
    (CASE WHEN CREDIT_SCORE < 580           THEN 1 ELSE 0 END +
     CASE WHEN DTI_RATIO > 43               THEN 1 ELSE 0 END +
     CASE WHEN LTV_RATIO > 95               THEN 1 ELSE 0 END +
     CASE WHEN NUM_DELINQUENCIES > 2        THEN 1 ELSE 0 END +
     CASE WHEN NUM_INQUIRIES_12M > 5        THEN 1 ELSE 0 END +
     CASE WHEN ENSEMBLE_DEFAULT_PROB > 0.50 THEN 1 ELSE 0 END +
     CASE WHEN IF_IS_ANOMALY = 1            THEN 1 ELSE 0 END) AS TOTAL_FLAGS
FROM risk_ranked
WHERE RISK_DECILE <= 3  -- Top 30% riskiest loans
ORDER BY RISK_SCORE DESC;

-- ============================================================
-- VIEW 7: Model Performance Dashboard
-- ============================================================
CREATE OR REPLACE VIEW VW_MODEL_PERFORMANCE AS
SELECT
    MODEL_NAME,
    MODEL_VERSION,
    EVAL_DATE,
    DATASET,
    N_SAMPLES,
    DEFAULT_RATE,
    ROUND(AUC_ROC, 4)                               AS AUC_ROC,
    ROUND(AUC_PR, 4)                                AS AUC_PR,
    ROUND(F1_SCORE, 4)                              AS F1_SCORE,
    ROUND(PRECISION_SCORE, 4)                       AS PRECISION_SCORE,
    ROUND(RECALL_SCORE, 4)                          AS RECALL_SCORE,
    ROUND(KS_STATISTIC, 4)                          AS KS_STATISTIC,
    ROUND(GINI_COEFFICIENT, 4)                      AS GINI_COEFFICIENT,
    -- Rank models by AUC within each dataset
    RANK() OVER (PARTITION BY DATASET ORDER BY AUC_ROC DESC) AS AUC_RANK
FROM ML.MODEL_PERFORMANCE
ORDER BY DATASET, AUC_ROC DESC;

-- ============================================================
-- VIEW 8: Sector-level default concentration
-- ============================================================
CREATE OR REPLACE VIEW VW_SECTOR_RISK AS
SELECT
    EMPLOYMENT_SECTOR,
    LOAN_TYPE,
    COUNT(*)                                        AS loan_count,
    ROUND(AVG(IS_DEFAULT) * 100, 2)                 AS default_rate_pct,
    ROUND(AVG(ANNUAL_INCOME), 0)                    AS avg_income,
    ROUND(AVG(CREDIT_SCORE), 0)                     AS avg_credit_score,
    ROUND(AVG(DTI_RATIO), 1)                        AS avg_dti,
    ROUND(SUM(LOAN_AMOUNT)/1000000, 2)              AS exposure_millions
FROM ANALYTICS.LOANS_CLEAN
GROUP BY EMPLOYMENT_SECTOR, LOAN_TYPE
ORDER BY default_rate_pct DESC;

-- Verify all views
SHOW VIEWS IN SCHEMA MERIDIAN_CREDIT_RISK.REPORTING;
