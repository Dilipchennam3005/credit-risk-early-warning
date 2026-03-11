"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
AWS Glue ETL Job: loan_data_transform
================================================================================
This script runs as an AWS Glue job. It:
  1. Reads raw partitioned CSVs from S3 (raw/loans/)
  2. Applies data quality checks and cleaning rules
  3. Engineers derived features for downstream ML
  4. Writes clean Parquet files back to S3 (processed/loans_clean/)
     partitioned by year and loan_type for efficient querying

Glue Job Parameters (passed at runtime):
  --JOB_NAME           : Glue job name
  --SOURCE_PATH        : s3://bucket/raw/loans/
  --TARGET_PATH        : s3://bucket/processed/loans_clean/
  --YEAR_FILTER        : optional — process only a specific year (e.g. 2023)

Author: Dilip Chennam
================================================================================
"""

import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# ── Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME', 'SOURCE_PATH', 'TARGET_PATH'
])

# ── Glue context setup
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args['JOB_NAME'], args)

SOURCE_PATH = args['SOURCE_PATH']
TARGET_PATH = args['TARGET_PATH']

logger.info(f"Starting ETL job: {args['JOB_NAME']}")
logger.info(f"Source: {SOURCE_PATH}")
logger.info(f"Target: {TARGET_PATH}")

# ================================================================================
# STEP 1 — READ RAW DATA
# ================================================================================
logger.info("Step 1: Reading raw CSV data from S3...")

df_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "false") \
    .csv(SOURCE_PATH)

raw_count = df_raw.count()
logger.info(f"  Raw records loaded: {raw_count:,}")

# ================================================================================
# STEP 2 — DATA QUALITY CHECKS
# ================================================================================
logger.info("Step 2: Running data quality checks...")

# Track rejected records
dq_rejects = []

# Rule 1: loan_amount must be positive
invalid_amount = df_raw.filter(F.col('loan_amount') <= 0).count()
if invalid_amount > 0:
    logger.warning(f"  DQ Rule 1 FAIL: {invalid_amount} records with loan_amount <= 0")
    dq_rejects.append(('invalid_loan_amount', invalid_amount))

# Rule 2: credit_score must be 300–850
invalid_score = df_raw.filter(
    (F.col('credit_score') < 300) | (F.col('credit_score') > 850)
).count()
if invalid_score > 0:
    logger.warning(f"  DQ Rule 2 FAIL: {invalid_score} records with out-of-range credit score")
    dq_rejects.append(('invalid_credit_score', invalid_score))

# Rule 3: dti_ratio must be 0–100
invalid_dti = df_raw.filter(
    (F.col('dti_ratio') < 0) | (F.col('dti_ratio') > 100)
).count()
if invalid_dti > 0:
    logger.warning(f"  DQ Rule 3 FAIL: {invalid_dti} records with invalid DTI")
    dq_rejects.append(('invalid_dti', invalid_dti))

# Rule 4: no null loan_id or loan_type
null_keys = df_raw.filter(
    F.col('loan_id').isNull() | F.col('loan_type').isNull()
).count()
if null_keys > 0:
    logger.warning(f"  DQ Rule 4 FAIL: {null_keys} records with null key fields")
    dq_rejects.append(('null_key_fields', null_keys))

# Rule 5: is_default must be 0 or 1
invalid_target = df_raw.filter(
    ~F.col('is_default').isin([0, 1])
).count()
if invalid_target > 0:
    logger.warning(f"  DQ Rule 5 FAIL: {invalid_target} records with invalid target variable")
    dq_rejects.append(('invalid_target', invalid_target))

logger.info(f"  DQ checks complete. Total issues: {sum(v for _, v in dq_rejects)}")

# Filter out invalid records
df_clean = df_raw \
    .filter(F.col('loan_amount') > 0) \
    .filter((F.col('credit_score') >= 300) & (F.col('credit_score') <= 850)) \
    .filter((F.col('dti_ratio') >= 0) & (F.col('dti_ratio') <= 100)) \
    .filter(F.col('loan_id').isNotNull()) \
    .filter(F.col('loan_type').isNotNull()) \
    .filter(F.col('is_default').isin([0, 1]))

clean_count = df_clean.count()
logger.info(f"  Records after DQ filtering: {clean_count:,} ({raw_count - clean_count:,} removed)")

# ================================================================================
# STEP 3 — DATA CLEANING & STANDARDIZATION
# ================================================================================
logger.info("Step 3: Cleaning and standardizing fields...")

df_clean = df_clean \
    .withColumn('origination_date', F.to_date('origination_date', 'yyyy-MM-dd')) \
    .withColumn('loan_type',        F.upper(F.trim(F.col('loan_type')))) \
    .withColumn('state',            F.upper(F.trim(F.col('state')))) \
    .withColumn('credit_tier',      F.trim(F.col('credit_tier'))) \
    .withColumn('loan_status',      F.trim(F.col('loan_status'))) \
    .withColumn('employment_sector',F.trim(F.col('employment_sector'))) \
    .withColumn('interest_rate',    F.round(F.col('interest_rate'), 3)) \
    .withColumn('dti_ratio',        F.round(F.col('dti_ratio'), 2)) \
    .withColumn('ltv_ratio',        F.round(F.col('ltv_ratio'), 2)) \
    .withColumn('annual_income',    F.round(F.col('annual_income'), 2)) \
    .withColumn('monthly_income',   F.round(F.col('monthly_income'), 2))

# Null handling
df_clean = df_clean \
    .withColumn('num_delinquencies', F.coalesce(F.col('num_delinquencies'), F.lit(0))) \
    .withColumn('num_inquiries_12m', F.coalesce(F.col('num_inquiries_12m'), F.lit(0))) \
    .withColumn('months_to_default', F.coalesce(F.col('months_to_default'), F.lit(-1)))

logger.info("  ✅ Cleaning complete")

# ================================================================================
# STEP 4 — FEATURE ENGINEERING
# ================================================================================
logger.info("Step 4: Engineering features for ML pipeline...")

df_features = df_clean \
    \
    .withColumn('loan_to_income_ratio',
        F.round(F.col('loan_amount') / F.col('annual_income'), 4)) \
    \
    .withColumn('payment_to_income_ratio',
        F.round(F.col('monthly_payment') / F.col('monthly_income'), 4)) \
    \
    .withColumn('credit_score_bucket',
        F.when(F.col('credit_score') >= 800, '800+')
         .when(F.col('credit_score') >= 740, '740-799')
         .when(F.col('credit_score') >= 670, '670-739')
         .when(F.col('credit_score') >= 620, '620-669')
         .when(F.col('credit_score') >= 580, '580-619')
         .otherwise('<580')) \
    \
    .withColumn('dti_bucket',
        F.when(F.col('dti_ratio') < 20, 'Low (<20)')
         .when(F.col('dti_ratio') < 36, 'Moderate (20-36)')
         .when(F.col('dti_ratio') < 43, 'High (36-43)')
         .otherwise('Very High (43+)')) \
    \
    .withColumn('ltv_bucket',
        F.when(F.col('ltv_ratio') < 60,  'Low (<60)')
         .when(F.col('ltv_ratio') < 80,  'Standard (60-80)')
         .when(F.col('ltv_ratio') < 90,  'High (80-90)')
         .when(F.col('ltv_ratio') < 97,  'Very High (90-97)')
         .otherwise('Above 97')) \
    \
    .withColumn('income_bucket',
        F.when(F.col('annual_income') < 40000,  'Low (<40K)')
         .when(F.col('annual_income') < 75000,  'Lower-Mid (40-75K)')
         .when(F.col('annual_income') < 120000, 'Middle (75-120K)')
         .when(F.col('annual_income') < 200000, 'Upper-Mid (120-200K)')
         .otherwise('High (200K+)')) \
    \
    .withColumn('rate_spread',
        F.round(F.col('interest_rate') - F.col('fed_funds_rate'), 3)) \
    \
    .withColumn('is_high_risk',
        F.when(
            (F.col('credit_score') < 620) |
            (F.col('dti_ratio') > 43) |
            (F.col('ltv_ratio') > 95) |
            (F.col('num_delinquencies') > 2),
            F.lit(1)
        ).otherwise(F.lit(0))) \
    \
    .withColumn('risk_score',
        F.round(
            (F.lit(850) - F.col('credit_score')) / F.lit(550) * F.lit(40) +
            F.col('dti_ratio') / F.lit(2) +
            F.col('ltv_ratio') / F.lit(10) +
            F.col('num_delinquencies') * F.lit(5) +
            F.col('num_inquiries_12m') * F.lit(2),
        2)) \
    \
    .withColumn('origination_month',  F.month('origination_date')) \
    .withColumn('origination_year',   F.year('origination_date')) \
    .withColumn('origination_quarter',
        F.concat(F.lit('Q'), F.ceil(F.month('origination_date') / 3).cast('string'))) \
    \
    .withColumn('is_covid_period',
        F.when(
            (F.year('origination_date') == 2020) &
            (F.month('origination_date').between(3, 12)),
            F.lit(1)
        ).otherwise(F.lit(0))) \
    \
    .withColumn('is_rate_hike_period',
        F.when(
            (F.year('origination_date') == 2022) |
            (F.year('origination_date') == 2023),
            F.lit(1)
        ).otherwise(F.lit(0))) \
    \
    .withColumn('etl_processed_at', F.current_timestamp()) \
    .withColumn('etl_job_name',     F.lit(args['JOB_NAME']))

logger.info(f"  ✅ Feature engineering complete. Total columns: {len(df_features.columns)}")

# ================================================================================
# STEP 5 — WRITE TO S3 AS PARQUET
# ================================================================================
logger.info("Step 5: Writing clean Parquet data to S3...")

df_features \
    .repartition(20, 'origination_year', 'loan_type') \
    .write \
    .mode('overwrite') \
    .partitionBy('origination_year', 'loan_type') \
    .parquet(TARGET_PATH)

logger.info(f"  ✅ Written to: {TARGET_PATH}")

# ================================================================================
# STEP 6 — JOB SUMMARY
# ================================================================================
logger.info("\n" + "=" * 60)
logger.info("ETL JOB SUMMARY")
logger.info("=" * 60)
logger.info(f"  Raw records:            {raw_count:,}")
logger.info(f"  Clean records written:  {clean_count:,}")
logger.info(f"  Records removed (DQ):   {raw_count - clean_count:,}")
logger.info(f"  Features engineered:    {len(df_features.columns)}")
logger.info(f"  Output path:            {TARGET_PATH}")
logger.info(f"  DQ issues found:        {dq_rejects}")
logger.info("=" * 60)

job.commit()
logger.info("Job committed successfully.")
