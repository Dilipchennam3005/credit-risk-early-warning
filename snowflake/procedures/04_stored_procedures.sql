-- ============================================================
-- Meridian Bank — Credit Risk Early Warning System
-- Snowflake: Stored Procedures
-- File: snowflake/procedures/04_stored_procedures.sql
-- Author: Dilip Chennam
-- ============================================================

USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
USE DATABASE MERIDIAN_CREDIT_RISK;
USE SCHEMA ANALYTICS;

-- ============================================================
-- PROCEDURE 1: Load model scores from ML pipeline output
-- ============================================================
CREATE OR REPLACE PROCEDURE ML.SP_LOAD_MODEL_SCORES(
    MODEL_VERSION VARCHAR,
    SCORES_STAGE  VARCHAR
)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
DECLARE
    v_inserted  INTEGER;
    v_start     TIMESTAMP_NTZ := CURRENT_TIMESTAMP();
BEGIN
    -- Create temp table for incoming scores
    CREATE OR REPLACE TEMPORARY TABLE TEMP_SCORES (
        LOAN_ID             VARCHAR(30),
        LR_DEFAULT_PROB     FLOAT,
        LR_PREDICTION       NUMBER(1),
        XGB_DEFAULT_PROB    FLOAT,
        XGB_PREDICTION      NUMBER(1),
        ENSEMBLE_DEFAULT_PROB FLOAT,
        ENSEMBLE_PREDICTION NUMBER(1),
        IF_ANOMALY_SCORE    FLOAT,
        IF_IS_ANOMALY       NUMBER(1)
    );

    -- Assign risk tier based on ensemble probability
    MERGE INTO ML.MODEL_SCORES tgt
    USING (
        SELECT
            t.LOAN_ID,
            :MODEL_VERSION                      AS MODEL_VERSION,
            CURRENT_TIMESTAMP()                 AS SCORED_AT,
            t.LR_DEFAULT_PROB,
            t.LR_PREDICTION,
            t.XGB_DEFAULT_PROB,
            t.XGB_PREDICTION,
            t.ENSEMBLE_DEFAULT_PROB,
            t.ENSEMBLE_PREDICTION,
            t.IF_ANOMALY_SCORE,
            t.IF_IS_ANOMALY,
            CASE
                WHEN t.ENSEMBLE_DEFAULT_PROB >= 0.60 THEN 'CRITICAL'
                WHEN t.ENSEMBLE_DEFAULT_PROB >= 0.40 THEN 'HIGH'
                WHEN t.ENSEMBLE_DEFAULT_PROB >= 0.20 THEN 'MEDIUM'
                ELSE 'LOW'
            END AS RISK_TIER
        FROM TEMP_SCORES t
    ) src ON (tgt.LOAN_ID = src.LOAN_ID AND tgt.MODEL_VERSION = src.MODEL_VERSION)
    WHEN MATCHED THEN UPDATE SET
        tgt.LR_DEFAULT_PROB       = src.LR_DEFAULT_PROB,
        tgt.XGB_DEFAULT_PROB      = src.XGB_DEFAULT_PROB,
        tgt.ENSEMBLE_DEFAULT_PROB = src.ENSEMBLE_DEFAULT_PROB,
        tgt.IF_ANOMALY_SCORE      = src.IF_ANOMALY_SCORE,
        tgt.IF_IS_ANOMALY         = src.IF_IS_ANOMALY,
        tgt.RISK_TIER             = src.RISK_TIER,
        tgt.SCORED_AT             = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT VALUES (
        src.LOAN_ID, src.MODEL_VERSION, src.SCORED_AT,
        src.LR_DEFAULT_PROB, src.LR_PREDICTION,
        src.XGB_DEFAULT_PROB, src.XGB_PREDICTION,
        src.ENSEMBLE_DEFAULT_PROB, src.ENSEMBLE_PREDICTION,
        src.IF_ANOMALY_SCORE, src.IF_IS_ANOMALY,
        src.RISK_TIER
    );

    SELECT COUNT(*) INTO v_inserted FROM ML.MODEL_SCORES WHERE MODEL_VERSION = :MODEL_VERSION;

    RETURN 'Loaded ' || v_inserted || ' scores for model version ' || :MODEL_VERSION ||
           ' in ' || DATEDIFF('second', :v_start, CURRENT_TIMESTAMP()) || 's';
END;
$$;


-- ============================================================
-- PROCEDURE 2: Log model evaluation metrics
-- ============================================================
CREATE OR REPLACE PROCEDURE ML.SP_LOG_MODEL_PERFORMANCE(
    P_MODEL_NAME        VARCHAR,
    P_MODEL_VERSION     VARCHAR,
    P_DATASET           VARCHAR,
    P_N_SAMPLES         INTEGER,
    P_N_DEFAULTS        INTEGER,
    P_AUC_ROC           FLOAT,
    P_AUC_PR            FLOAT,
    P_ACCURACY          FLOAT,
    P_PRECISION         FLOAT,
    P_RECALL            FLOAT,
    P_F1                FLOAT,
    P_KS                FLOAT
)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
DECLARE
    v_gini FLOAT;
    v_default_rate FLOAT;
BEGIN
    v_gini         := 2 * :P_AUC_ROC - 1;
    v_default_rate := :P_N_DEFAULTS / NULLIF(:P_N_SAMPLES, 0);

    MERGE INTO ML.MODEL_PERFORMANCE tgt
    USING (SELECT
        :P_MODEL_NAME    AS MODEL_NAME,
        :P_MODEL_VERSION AS MODEL_VERSION,
        :P_DATASET       AS DATASET
    ) src
    ON (tgt.MODEL_NAME    = src.MODEL_NAME
    AND tgt.MODEL_VERSION = src.MODEL_VERSION
    AND tgt.DATASET       = src.DATASET)
    WHEN MATCHED THEN UPDATE SET
        EVAL_DATE        = CURRENT_DATE(),
        N_SAMPLES        = :P_N_SAMPLES,
        N_DEFAULTS       = :P_N_DEFAULTS,
        DEFAULT_RATE     = :v_default_rate,
        AUC_ROC          = :P_AUC_ROC,
        AUC_PR           = :P_AUC_PR,
        ACCURACY         = :P_ACCURACY,
        PRECISION_SCORE  = :P_PRECISION,
        RECALL_SCORE     = :P_RECALL,
        F1_SCORE         = :P_F1,
        KS_STATISTIC     = :P_KS,
        GINI_COEFFICIENT = :v_gini
    WHEN NOT MATCHED THEN INSERT (
        MODEL_NAME, MODEL_VERSION, EVAL_DATE, DATASET,
        N_SAMPLES, N_DEFAULTS, DEFAULT_RATE,
        AUC_ROC, AUC_PR, ACCURACY, PRECISION_SCORE,
        RECALL_SCORE, F1_SCORE, KS_STATISTIC, GINI_COEFFICIENT
    ) VALUES (
        :P_MODEL_NAME, :P_MODEL_VERSION, CURRENT_DATE(), :P_DATASET,
        :P_N_SAMPLES, :P_N_DEFAULTS, :v_default_rate,
        :P_AUC_ROC, :P_AUC_PR, :P_ACCURACY, :P_PRECISION,
        :P_RECALL, :P_F1, :P_KS, :v_gini
    );

    RETURN 'Logged performance for ' || :P_MODEL_NAME || ' v' || :P_MODEL_VERSION ||
           ' (' || :P_DATASET || '): AUC=' || :P_AUC_ROC || ', F1=' || :P_F1;
END;
$$;


-- ============================================================
-- PROCEDURE 3: Daily risk tier refresh
-- ============================================================
CREATE OR REPLACE PROCEDURE ANALYTICS.SP_REFRESH_RISK_TIERS()
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
DECLARE
    v_updated   INTEGER;
    v_critical  INTEGER;
    v_high      INTEGER;
    v_medium    INTEGER;
    v_low       INTEGER;
BEGIN
    -- Update risk tiers in model scores based on latest ensemble probability
    UPDATE ML.MODEL_SCORES
    SET RISK_TIER = CASE
        WHEN ENSEMBLE_DEFAULT_PROB >= 0.60 THEN 'CRITICAL'
        WHEN ENSEMBLE_DEFAULT_PROB >= 0.40 THEN 'HIGH'
        WHEN ENSEMBLE_DEFAULT_PROB >= 0.20 THEN 'MEDIUM'
        ELSE 'LOW'
    END;

    SELECT COUNT(*) INTO v_updated  FROM ML.MODEL_SCORES;
    SELECT COUNT(*) INTO v_critical FROM ML.MODEL_SCORES WHERE RISK_TIER = 'CRITICAL';
    SELECT COUNT(*) INTO v_high     FROM ML.MODEL_SCORES WHERE RISK_TIER = 'HIGH';
    SELECT COUNT(*) INTO v_medium   FROM ML.MODEL_SCORES WHERE RISK_TIER = 'MEDIUM';
    SELECT COUNT(*) INTO v_low      FROM ML.MODEL_SCORES WHERE RISK_TIER = 'LOW';

    RETURN 'Risk tiers refreshed for ' || v_updated || ' loans. ' ||
           'CRITICAL: ' || v_critical || ' | HIGH: ' || v_high ||
           ' | MEDIUM: ' || v_medium || ' | LOW: ' || v_low;
END;
$$;


-- ============================================================
-- PROCEDURE 4: Generate portfolio risk summary report
-- ============================================================
CREATE OR REPLACE PROCEDURE REPORTING.SP_GENERATE_RISK_REPORT(
    P_AS_OF_DATE DATE DEFAULT CURRENT_DATE()
)
RETURNS TABLE (METRIC VARCHAR, VALUE VARCHAR)
LANGUAGE SQL
AS
$$
BEGIN
    RETURN TABLE(
        SELECT 'Report Date',               :P_AS_OF_DATE::VARCHAR          UNION ALL
        SELECT 'Total Loans',               COUNT(*)::VARCHAR
            FROM ANALYTICS.LOANS_CLEAN      UNION ALL
        SELECT 'Total Exposure ($M)',        ROUND(SUM(LOAN_AMOUNT)/1000000,2)::VARCHAR
            FROM ANALYTICS.LOANS_CLEAN      UNION ALL
        SELECT 'Overall Default Rate (%)',   ROUND(AVG(IS_DEFAULT)*100,2)::VARCHAR
            FROM ANALYTICS.LOANS_CLEAN      UNION ALL
        SELECT 'High Risk Loans',            SUM(IS_HIGH_RISK)::VARCHAR
            FROM ANALYTICS.LOANS_CLEAN      UNION ALL
        SELECT 'Avg Credit Score',           ROUND(AVG(CREDIT_SCORE),0)::VARCHAR
            FROM ANALYTICS.LOANS_CLEAN      UNION ALL
        SELECT 'Avg DTI Ratio (%)',          ROUND(AVG(DTI_RATIO),1)::VARCHAR
            FROM ANALYTICS.LOANS_CLEAN      UNION ALL
        SELECT 'Critical Risk Loans',        COUNT(*)::VARCHAR
            FROM ML.MODEL_SCORES WHERE RISK_TIER = 'CRITICAL'
    );
END;
$$;


-- ============================================================
-- Test the procedures
-- ============================================================

-- Log sample model performance
CALL ML.SP_LOG_MODEL_PERFORMANCE(
    'XGBoost', 'v1.0', 'TEST',
    50000, 5512, 0.9421, 0.8103,
    0.9124, 0.8234, 0.7891, 0.8045, 0.7234
);

CALL ML.SP_LOG_MODEL_PERFORMANCE(
    'LogisticRegression', 'v1.0', 'TEST',
    50000, 5512, 0.8912, 0.7543,
    0.8934, 0.7823, 0.7234, 0.7519, 0.6821
);

CALL ML.SP_LOG_MODEL_PERFORMANCE(
    'XGBoost', 'v1.0', 'TRAIN',
    200000, 22048, 0.9634, 0.8891,
    0.9312, 0.8734, 0.8123, 0.8417, 0.7892
);

-- Verify
SELECT * FROM ML.MODEL_PERFORMANCE ORDER BY MODEL_NAME, DATASET;

-- Generate report
CALL REPORTING.SP_GENERATE_RISK_REPORT();
