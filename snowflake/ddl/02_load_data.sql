-- ============================================================
-- Meridian Bank — Credit Risk Early Warning System
-- Snowflake: Data Loading from S3
-- File: snowflake/ddl/02_load_data.sql
-- Author: Dilip Chennam
-- ============================================================

USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
USE DATABASE MERIDIAN_CREDIT_RISK;
USE SCHEMA RAW;

-- ── Load raw loan data directly from CSV
-- Since we are loading from local CSV (not S3 stage in free tier),
-- use Snowflake's web UI: Data → Load Data → select loans_raw.csv
-- Or use snowsql CLI:

-- snowsql -a <account> -u <user> -d MERIDIAN_CREDIT_RISK -s RAW
-- PUT file://data/raw/loans_raw.csv @~/staged_loans AUTO_COMPRESS=TRUE;
-- COPY INTO RAW.LOANS_RAW FROM @~/staged_loans/loans_raw.csv.gz FILE_FORMAT = CSV_FORMAT;

-- ── Alternative: Load via Snowflake internal stage
CREATE OR REPLACE STAGE INTERNAL_STAGE
    FILE_FORMAT = CSV_FORMAT
    COMMENT = 'Internal stage for local CSV uploads';

-- After PUT command loads the file, run COPY INTO:
COPY INTO RAW.LOANS_RAW (
    LOAN_ID, LOAN_TYPE, ORIGINATION_DATE, ORIGINATION_YEAR, ORIGINATION_QUARTER,
    LOAN_AMOUNT, TERM_MONTHS, INTEREST_RATE, LOAN_PURPOSE, LTV_RATIO, MONTHLY_PAYMENT,
    BORROWER_AGE, ANNUAL_INCOME, MONTHLY_INCOME, EMPLOYMENT_YEARS, IS_SELF_EMPLOYED,
    EMPLOYMENT_SECTOR, STATE, CREDIT_SCORE, CREDIT_TIER, DTI_RATIO,
    NUM_OPEN_ACCOUNTS, NUM_DELINQUENCIES, NUM_INQUIRIES_12M, FED_FUNDS_RATE,
    PROPERTY_TYPE, PROPERTY_VALUE, VEHICLE_MAKE, VEHICLE_MODEL, VEHICLE_YEAR,
    VEHICLE_TYPE, LOAN_STATUS, IS_DEFAULT, DEFAULT_PROBABILITY, MONTHS_TO_DEFAULT
)
FROM @INTERNAL_STAGE/loans_raw.csv.gz
FILE_FORMAT = CSV_FORMAT
ON_ERROR = 'CONTINUE'
PURGE = FALSE;

-- Verify load
SELECT
    COUNT(*)                                    AS total_records,
    SUM(IS_DEFAULT)                             AS total_defaults,
    ROUND(AVG(IS_DEFAULT) * 100, 2)             AS default_rate_pct,
    COUNT(DISTINCT LOAN_TYPE)                   AS loan_types,
    MIN(ORIGINATION_DATE)                       AS earliest_loan,
    MAX(ORIGINATION_DATE)                       AS latest_loan
FROM RAW.LOANS_RAW;

-- ── Transform raw → analytics (feature engineering in SQL)
USE SCHEMA ANALYTICS;

INSERT INTO ANALYTICS.LOANS_CLEAN
SELECT
    -- Raw fields
    r.LOAN_ID, r.LOAN_TYPE, r.ORIGINATION_DATE, r.ORIGINATION_YEAR,
    r.ORIGINATION_QUARTER, MONTH(r.ORIGINATION_DATE) AS ORIGINATION_MONTH,
    r.LOAN_AMOUNT, r.TERM_MONTHS, r.INTEREST_RATE, r.LOAN_PURPOSE,
    r.LTV_RATIO, r.MONTHLY_PAYMENT, r.BORROWER_AGE, r.ANNUAL_INCOME,
    r.MONTHLY_INCOME, r.EMPLOYMENT_YEARS, r.IS_SELF_EMPLOYED,
    r.EMPLOYMENT_SECTOR, r.STATE, r.CREDIT_SCORE, r.CREDIT_TIER,
    r.DTI_RATIO, r.NUM_OPEN_ACCOUNTS, r.NUM_DELINQUENCIES,
    r.NUM_INQUIRIES_12M, r.FED_FUNDS_RATE, r.PROPERTY_TYPE,
    r.PROPERTY_VALUE, r.VEHICLE_MAKE, r.VEHICLE_MODEL, r.VEHICLE_YEAR,
    r.VEHICLE_TYPE, r.LOAN_STATUS, r.IS_DEFAULT, r.DEFAULT_PROBABILITY,
    COALESCE(r.MONTHS_TO_DEFAULT, -1),

    -- Engineered features
    ROUND(r.LOAN_AMOUNT / NULLIF(r.ANNUAL_INCOME, 0), 4)            AS LOAN_TO_INCOME_RATIO,
    ROUND(r.MONTHLY_PAYMENT / NULLIF(r.MONTHLY_INCOME, 0), 4)       AS PAYMENT_TO_INCOME_RATIO,
    ROUND(r.INTEREST_RATE - r.FED_FUNDS_RATE, 3)                    AS RATE_SPREAD,

    -- Risk score (composite)
    ROUND(
        (850 - r.CREDIT_SCORE) / 550.0 * 40 +
        r.DTI_RATIO / 2.0 +
        r.LTV_RATIO / 10.0 +
        r.NUM_DELINQUENCIES * 5 +
        r.NUM_INQUIRIES_12M * 2,
    2)                                                               AS RISK_SCORE,

    -- High risk flag
    CASE WHEN r.CREDIT_SCORE < 620
          OR r.DTI_RATIO > 43
          OR r.LTV_RATIO > 95
          OR r.NUM_DELINQUENCIES > 2
         THEN 1 ELSE 0 END                                           AS IS_HIGH_RISK,

    -- Buckets
    CASE
        WHEN r.CREDIT_SCORE >= 800 THEN '800+'
        WHEN r.CREDIT_SCORE >= 740 THEN '740-799'
        WHEN r.CREDIT_SCORE >= 670 THEN '670-739'
        WHEN r.CREDIT_SCORE >= 620 THEN '620-669'
        WHEN r.CREDIT_SCORE >= 580 THEN '580-619'
        ELSE '<580'
    END                                                              AS CREDIT_SCORE_BUCKET,

    CASE
        WHEN r.DTI_RATIO < 20  THEN 'Low (<20)'
        WHEN r.DTI_RATIO < 36  THEN 'Moderate (20-36)'
        WHEN r.DTI_RATIO < 43  THEN 'High (36-43)'
        ELSE 'Very High (43+)'
    END                                                              AS DTI_BUCKET,

    CASE
        WHEN r.LTV_RATIO < 60  THEN 'Low (<60)'
        WHEN r.LTV_RATIO < 80  THEN 'Standard (60-80)'
        WHEN r.LTV_RATIO < 90  THEN 'High (80-90)'
        WHEN r.LTV_RATIO < 97  THEN 'Very High (90-97)'
        ELSE 'Above 97'
    END                                                              AS LTV_BUCKET,

    CASE
        WHEN r.ANNUAL_INCOME < 40000  THEN 'Low (<40K)'
        WHEN r.ANNUAL_INCOME < 75000  THEN 'Lower-Mid (40-75K)'
        WHEN r.ANNUAL_INCOME < 120000 THEN 'Middle (75-120K)'
        WHEN r.ANNUAL_INCOME < 200000 THEN 'Upper-Mid (120-200K)'
        ELSE 'High (200K+)'
    END                                                              AS INCOME_BUCKET,

    CASE WHEN r.ORIGINATION_YEAR = 2020
          AND MONTH(r.ORIGINATION_DATE) >= 3
         THEN 1 ELSE 0 END                                           AS IS_COVID_PERIOD,

    CASE WHEN r.ORIGINATION_YEAR IN (2022, 2023)
         THEN 1 ELSE 0 END                                           AS IS_RATE_HIKE_PERIOD,

    CURRENT_TIMESTAMP()                                              AS PROCESSED_AT

FROM RAW.LOANS_RAW r
WHERE r.LOAN_AMOUNT > 0
  AND r.CREDIT_SCORE BETWEEN 300 AND 850
  AND r.DTI_RATIO BETWEEN 0 AND 100
  AND r.IS_DEFAULT IN (0, 1);

-- Verify
SELECT
    LOAN_TYPE,
    COUNT(*)                        AS records,
    ROUND(AVG(IS_DEFAULT)*100, 2)   AS default_rate_pct,
    ROUND(AVG(CREDIT_SCORE), 0)     AS avg_credit_score,
    ROUND(AVG(DTI_RATIO), 1)        AS avg_dti,
    ROUND(AVG(RISK_SCORE), 1)       AS avg_risk_score,
    SUM(IS_HIGH_RISK)               AS high_risk_count
FROM ANALYTICS.LOANS_CLEAN
GROUP BY LOAN_TYPE;
