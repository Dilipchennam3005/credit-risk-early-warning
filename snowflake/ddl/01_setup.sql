-- ============================================================
-- Meridian Bank — Credit Risk Early Warning System
-- Snowflake DDL: Database, Schema, and Tables
-- File: snowflake/ddl/01_setup.sql
-- Run as: ACCOUNTADMIN
-- Author: Dilip Chennam
-- ============================================================

-- ── Environment setup
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;

-- ── Database and schemas
CREATE DATABASE IF NOT EXISTS MERIDIAN_CREDIT_RISK
    COMMENT = 'Meridian Bank Credit Risk Early Warning System';

CREATE SCHEMA IF NOT EXISTS MERIDIAN_CREDIT_RISK.RAW
    COMMENT = 'Raw ingested loan data from S3';

CREATE SCHEMA IF NOT EXISTS MERIDIAN_CREDIT_RISK.ANALYTICS
    COMMENT = 'Cleaned, enriched, and feature-engineered loan data';

CREATE SCHEMA IF NOT EXISTS MERIDIAN_CREDIT_RISK.ML
    COMMENT = 'ML model scores, predictions, and risk tiers';

CREATE SCHEMA IF NOT EXISTS MERIDIAN_CREDIT_RISK.REPORTING
    COMMENT = 'Aggregated views for Tableau dashboards';

USE DATABASE MERIDIAN_CREDIT_RISK;

-- ============================================================
-- RAW SCHEMA — Landing tables for S3 ingestion
-- ============================================================
USE SCHEMA RAW;

CREATE OR REPLACE TABLE RAW.LOANS_RAW (
    -- Identifiers
    LOAN_ID                 VARCHAR(30)     NOT NULL,
    LOAN_TYPE               VARCHAR(10)     NOT NULL,
    ORIGINATION_DATE        DATE,
    ORIGINATION_YEAR        NUMBER(4),
    ORIGINATION_QUARTER     VARCHAR(2),

    -- Loan terms
    LOAN_AMOUNT             FLOAT,
    TERM_MONTHS             NUMBER(3),
    INTEREST_RATE           FLOAT,
    LOAN_PURPOSE            VARCHAR(30),
    LTV_RATIO               FLOAT,
    MONTHLY_PAYMENT         FLOAT,

    -- Borrower
    BORROWER_AGE            NUMBER(3),
    ANNUAL_INCOME           FLOAT,
    MONTHLY_INCOME          FLOAT,
    EMPLOYMENT_YEARS        FLOAT,
    IS_SELF_EMPLOYED        NUMBER(1),
    EMPLOYMENT_SECTOR       VARCHAR(50),
    STATE                   VARCHAR(2),

    -- Credit profile
    CREDIT_SCORE            NUMBER(3),
    CREDIT_TIER             VARCHAR(20),
    DTI_RATIO               FLOAT,
    NUM_OPEN_ACCOUNTS       NUMBER(3),
    NUM_DELINQUENCIES       NUMBER(3),
    NUM_INQUIRIES_12M       NUMBER(3),

    -- Economic context
    FED_FUNDS_RATE          FLOAT,

    -- Mortgage specific
    PROPERTY_TYPE           VARCHAR(30),
    PROPERTY_VALUE          FLOAT,

    -- Auto specific
    VEHICLE_MAKE            VARCHAR(30),
    VEHICLE_MODEL           VARCHAR(30),
    VEHICLE_YEAR            NUMBER(4),
    VEHICLE_TYPE            VARCHAR(30),

    -- Target
    LOAN_STATUS             VARCHAR(20),
    IS_DEFAULT              NUMBER(1),
    DEFAULT_PROBABILITY     FLOAT,
    MONTHS_TO_DEFAULT       NUMBER(3),

    -- Metadata
    INGESTED_AT             TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    SOURCE_FILE             VARCHAR(500),

    PRIMARY KEY (LOAN_ID)
)
COMMENT = 'Raw loan data ingested from S3 — mortgage and auto loans 2018-2024';

-- ============================================================
-- ANALYTICS SCHEMA — Cleaned and enriched tables
-- ============================================================
USE SCHEMA ANALYTICS;

CREATE OR REPLACE TABLE ANALYTICS.LOANS_CLEAN (
    -- All raw fields (cleaned)
    LOAN_ID                 VARCHAR(30)     NOT NULL,
    LOAN_TYPE               VARCHAR(10)     NOT NULL,
    ORIGINATION_DATE        DATE            NOT NULL,
    ORIGINATION_YEAR        NUMBER(4)       NOT NULL,
    ORIGINATION_QUARTER     VARCHAR(2),
    ORIGINATION_MONTH       NUMBER(2),
    LOAN_AMOUNT             FLOAT           NOT NULL,
    TERM_MONTHS             NUMBER(3)       NOT NULL,
    INTEREST_RATE           FLOAT           NOT NULL,
    LOAN_PURPOSE            VARCHAR(30),
    LTV_RATIO               FLOAT,
    MONTHLY_PAYMENT         FLOAT,
    BORROWER_AGE            NUMBER(3),
    ANNUAL_INCOME           FLOAT,
    MONTHLY_INCOME          FLOAT,
    EMPLOYMENT_YEARS        FLOAT,
    IS_SELF_EMPLOYED        NUMBER(1),
    EMPLOYMENT_SECTOR       VARCHAR(50),
    STATE                   VARCHAR(2),
    CREDIT_SCORE            NUMBER(3)       NOT NULL,
    CREDIT_TIER             VARCHAR(20),
    DTI_RATIO               FLOAT,
    NUM_OPEN_ACCOUNTS       NUMBER(3),
    NUM_DELINQUENCIES       NUMBER(3),
    NUM_INQUIRIES_12M       NUMBER(3),
    FED_FUNDS_RATE          FLOAT,
    PROPERTY_TYPE           VARCHAR(30),
    PROPERTY_VALUE          FLOAT,
    VEHICLE_MAKE            VARCHAR(30),
    VEHICLE_MODEL           VARCHAR(30),
    VEHICLE_YEAR            NUMBER(4),
    VEHICLE_TYPE            VARCHAR(30),
    LOAN_STATUS             VARCHAR(20),
    IS_DEFAULT              NUMBER(1)       NOT NULL,
    DEFAULT_PROBABILITY     FLOAT,
    MONTHS_TO_DEFAULT       NUMBER(3),

    -- Engineered features
    LOAN_TO_INCOME_RATIO    FLOAT,
    PAYMENT_TO_INCOME_RATIO FLOAT,
    RATE_SPREAD             FLOAT,
    RISK_SCORE              FLOAT,
    IS_HIGH_RISK            NUMBER(1),
    CREDIT_SCORE_BUCKET     VARCHAR(10),
    DTI_BUCKET              VARCHAR(20),
    LTV_BUCKET              VARCHAR(20),
    INCOME_BUCKET           VARCHAR(25),
    IS_COVID_PERIOD         NUMBER(1),
    IS_RATE_HIKE_PERIOD     NUMBER(1),

    -- Metadata
    PROCESSED_AT            TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (LOAN_ID)
)
CLUSTER BY (ORIGINATION_YEAR, LOAN_TYPE)
COMMENT = 'Cleaned and feature-engineered loan data — source for ML and reporting';

-- ============================================================
-- ML SCHEMA — Model scores and predictions
-- ============================================================
USE SCHEMA ML;

CREATE OR REPLACE TABLE ML.MODEL_SCORES (
    LOAN_ID                 VARCHAR(30)     NOT NULL,
    MODEL_VERSION           VARCHAR(20)     NOT NULL,
    SCORED_AT               TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),

    -- Logistic Regression
    LR_DEFAULT_PROB         FLOAT,
    LR_PREDICTION           NUMBER(1),

    -- XGBoost
    XGB_DEFAULT_PROB        FLOAT,
    XGB_PREDICTION          NUMBER(1),

    -- Ensemble (average of LR + XGB)
    ENSEMBLE_DEFAULT_PROB   FLOAT,
    ENSEMBLE_PREDICTION     NUMBER(1),

    -- Isolation Forest anomaly score
    IF_ANOMALY_SCORE        FLOAT,
    IF_IS_ANOMALY           NUMBER(1),

    -- Risk tier based on ensemble probability
    RISK_TIER               VARCHAR(10),    -- LOW / MEDIUM / HIGH / CRITICAL

    PRIMARY KEY (LOAN_ID, MODEL_VERSION)
)
COMMENT = 'ML model scores and risk tier assignments for all loans';

CREATE OR REPLACE TABLE ML.MODEL_PERFORMANCE (
    MODEL_NAME              VARCHAR(50)     NOT NULL,
    MODEL_VERSION           VARCHAR(20)     NOT NULL,
    EVAL_DATE               DATE            DEFAULT CURRENT_DATE(),
    DATASET                 VARCHAR(10),    -- TRAIN / TEST / VALIDATION
    N_SAMPLES               NUMBER,
    N_DEFAULTS              NUMBER,
    DEFAULT_RATE            FLOAT,
    AUC_ROC                 FLOAT,
    AUC_PR                  FLOAT,
    ACCURACY                FLOAT,
    PRECISION_SCORE         FLOAT,
    RECALL_SCORE            FLOAT,
    F1_SCORE                FLOAT,
    KS_STATISTIC            FLOAT,
    GINI_COEFFICIENT        FLOAT,
    PRIMARY KEY (MODEL_NAME, MODEL_VERSION, DATASET)
)
COMMENT = 'Model evaluation metrics tracked across versions';

-- ============================================================
-- FILE FORMAT & STAGE for S3 loading
-- ============================================================
USE SCHEMA RAW;

CREATE OR REPLACE FILE FORMAT CSV_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    RECORD_DELIMITER = '\n'
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('NULL', 'null', 'NA', '')
    EMPTY_FIELD_AS_NULL = TRUE
    COMMENT = 'Standard CSV format for loan data ingestion';

CREATE OR REPLACE STAGE S3_LOAN_STAGE
    URL = 's3://meridian-credit-risk-354561615261/processed/loans_clean/'
    CREDENTIALS = (
        AWS_KEY_ID     = '${AWS_KEY_ID}'
        AWS_SECRET_KEY = '${AWS_SECRET_KEY}'
    )
    FILE_FORMAT = CSV_FORMAT
    COMMENT = 'External stage pointing to processed S3 data';

-- Verify stage
-- LIST @S3_LOAN_STAGE;

SHOW SCHEMAS IN DATABASE MERIDIAN_CREDIT_RISK;
SHOW TABLES IN SCHEMA MERIDIAN_CREDIT_RISK.RAW;
SHOW TABLES IN SCHEMA MERIDIAN_CREDIT_RISK.ANALYTICS;
SHOW TABLES IN SCHEMA MERIDIAN_CREDIT_RISK.ML;
