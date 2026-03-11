"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
AWS S3 Bucket Setup Script
================================================================================
Creates the S3 bucket structure for the data pipeline:

  meridian-credit-risk-{account_id}/
  ├── raw/
  │   ├── loans/year=2018/loan_type=Mortgage/
  │   ├── loans/year=2018/loan_type=Auto/
  │   └── ... (partitioned by year and loan_type)
  ├── processed/
  │   └── loans_clean/ (Parquet, partitioned — output of Glue ETL)
  ├── scripts/
  │   └── glue_etl_job.py (uploaded here for Glue to reference)
  └── logs/
      └── glue/ (Glue job logs)

Run this script ONCE to provision the S3 infrastructure.

Author: Dilip Chennam
================================================================================
"""

import boto3
import os
import json
from pathlib import Path

REGION      = 'us-west-2'
ACCOUNT_ID  = boto3.client('sts').get_caller_identity()['Account']
BUCKET_NAME = f'meridian-credit-risk-{ACCOUNT_ID}'

s3  = boto3.client('s3', region_name=REGION)
s3r = boto3.resource('s3', region_name=REGION)

def create_bucket():
    print(f"Creating S3 bucket: {BUCKET_NAME}")
    try:
        s3.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={'LocationConstraint': REGION}
        )
        print(f"  ✅ Bucket created: s3://{BUCKET_NAME}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  ℹ️  Bucket already exists: s3://{BUCKET_NAME}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise

def configure_bucket():
    print("\nConfiguring bucket settings...")

    # Block all public access
    s3.put_public_access_block(
        Bucket=BUCKET_NAME,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls':       True,
            'IgnorePublicAcls':      True,
            'BlockPublicPolicy':     True,
            'RestrictPublicBuckets': True,
        }
    )
    print("  ✅ Public access blocked")

    # Enable versioning
    s3.put_bucket_versioning(
        Bucket=BUCKET_NAME,
        VersioningConfiguration={'Status': 'Enabled'}
    )
    print("  ✅ Versioning enabled")

    # Enable server-side encryption (AES-256)
    s3.put_bucket_encryption(
        Bucket=BUCKET_NAME,
        ServerSideEncryptionConfiguration={
            'Rules': [{
                'ApplyServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'AES256'
                }
            }]
        }
    )
    print("  ✅ Server-side encryption enabled (AES-256)")

    # Lifecycle policy: move processed data to Glacier after 90 days
    s3.put_bucket_lifecycle_configuration(
        Bucket=BUCKET_NAME,
        LifecycleConfiguration={
            'Rules': [
                {
                    'ID':     'archive-processed-data',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'processed/'},
                    'Transitions': [{
                        'Days':         90,
                        'StorageClass': 'GLACIER'
                    }]
                },
                {
                    'ID':     'expire-logs',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'logs/'},
                    'Expiration': {'Days': 30}
                }
            ]
        }
    )
    print("  ✅ Lifecycle policy set (processed → Glacier after 90d, logs expire after 30d)")

def create_folder_structure():
    print("\nCreating folder structure...")
    prefixes = [
        'raw/loans/',
        'processed/loans_clean/',
        'scripts/',
        'logs/glue/',
    ]
    for prefix in prefixes:
        s3.put_object(Bucket=BUCKET_NAME, Key=prefix)
        print(f"  ✅ s3://{BUCKET_NAME}/{prefix}")

def upload_raw_data():
    print("\nUploading raw loan data (partitioned by year and loan_type)...")
    import pandas as pd

    raw_path = Path('data/raw/loans_raw.csv')
    if not raw_path.exists():
        print("  ⚠️  data/raw/loans_raw.csv not found — run generate_loan_data.py first")
        return

    print("  Loading CSV...")
    df = pd.read_csv(raw_path)

    uploaded = 0
    for (year, loan_type), group in df.groupby(['origination_year', 'loan_type']):
        key      = f'raw/loans/year={year}/loan_type={loan_type}/loans_{year}_{loan_type.lower()}.csv'
        csv_body = group.to_csv(index=False)
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=csv_body.encode('utf-8'),
            ContentType='text/csv'
        )
        print(f"  ✅ Uploaded {len(group):>6,} rows → s3://{BUCKET_NAME}/{key}")
        uploaded += len(group)

    print(f"\n  Total uploaded: {uploaded:,} rows across {df['origination_year'].nunique()} years × 2 loan types")

def upload_glue_script():
    print("\nUploading Glue ETL script to S3...")
    script_path = Path('aws/glue/glue_etl_job.py')
    if not script_path.exists():
        print("  ⚠️  aws/glue/glue_etl_job.py not found — skipping")
        return
    with open(script_path, 'rb') as f:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key='scripts/glue_etl_job.py',
            Body=f.read()
        )
    print(f"  ✅ s3://{BUCKET_NAME}/scripts/glue_etl_job.py")

def save_config():
    config = {
        'bucket_name':  BUCKET_NAME,
        'region':       REGION,
        'account_id':   ACCOUNT_ID,
        'paths': {
            'raw':       f's3://{BUCKET_NAME}/raw/loans/',
            'processed': f's3://{BUCKET_NAME}/processed/loans_clean/',
            'scripts':   f's3://{BUCKET_NAME}/scripts/',
            'logs':      f's3://{BUCKET_NAME}/logs/glue/',
        }
    }
    os.makedirs('aws/s3_configs', exist_ok=True)
    with open('aws/s3_configs/bucket_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  ✅ Config saved to aws/s3_configs/bucket_config.json")
    return config

if __name__ == '__main__':
    print("=" * 60)
    print("Meridian Bank — S3 Infrastructure Setup")
    print("=" * 60)

    create_bucket()
    configure_bucket()
    create_folder_structure()
    upload_raw_data()
    upload_glue_script()
    config = save_config()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"Bucket:    s3://{BUCKET_NAME}")
    print(f"Region:    {REGION}")
    print(f"Raw data:  {config['paths']['raw']}")
    print(f"Processed: {config['paths']['processed']}")
