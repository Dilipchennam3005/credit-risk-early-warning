"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
AWS Glue Job Provisioner
================================================================================
Creates and configures the AWS Glue ETL job via boto3.
Run this AFTER setup_s3.py to register the job in AWS Glue.

Author: Dilip Chennam
================================================================================
"""

import boto3
import json
import time

REGION = 'us-west-2'

sts        = boto3.client('sts', region_name=REGION)
glue       = boto3.client('glue', region_name=REGION)
iam        = boto3.client('iam', region_name=REGION)

ACCOUNT_ID  = sts.get_caller_identity()['Account']
BUCKET_NAME = f'meridian-credit-risk-{ACCOUNT_ID}'
JOB_NAME    = 'meridian-loan-data-transform'
ROLE_NAME   = 'MeridianGlueServiceRole'

def create_glue_iam_role():
    """Create IAM role for Glue with S3 access."""
    print(f"Creating IAM role: {ROLE_NAME}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "glue.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    try:
        response = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for Meridian Bank Glue ETL jobs'
        )
        role_arn = response['Role']['Arn']
        print(f"  ✅ Role created: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
        print(f"  ℹ️  Role already exists: {role_arn}")

    # Attach AWS managed policies
    policies = [
        'arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole',
        'arn:aws:iam::aws:policy/AmazonS3FullAccess',
        'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess',
    ]
    for policy in policies:
        try:
            iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy)
            print(f"  ✅ Attached: {policy.split('/')[-1]}")
        except Exception as e:
            print(f"  ℹ️  {policy.split('/')[-1]}: {e}")

    print("  Waiting 10s for IAM role to propagate...")
    time.sleep(10)
    return role_arn

def create_glue_database():
    """Create Glue Data Catalog database."""
    print("\nCreating Glue Data Catalog database...")
    try:
        glue.create_database(
            DatabaseInput={
                'Name':        'meridian_credit_risk',
                'Description': 'Meridian Bank Credit Risk Early Warning System'
            }
        )
        print("  ✅ Database created: meridian_credit_risk")
    except glue.exceptions.AlreadyExistsException:
        print("  ℹ️  Database already exists: meridian_credit_risk")

def create_glue_job(role_arn):
    """Create the Glue ETL job."""
    print(f"\nCreating Glue job: {JOB_NAME}")

    job_config = {
        'Name':        JOB_NAME,
        'Description': 'ETL job to clean, transform, and feature-engineer Meridian Bank loan data',
        'Role':        role_arn,
        'Command': {
            'Name':           'glueetl',
            'ScriptLocation': f's3://{BUCKET_NAME}/scripts/glue_etl_job.py',
            'PythonVersion':  '3'
        },
        'DefaultArguments': {
            '--JOB_NAME':         JOB_NAME,
            '--SOURCE_PATH':      f's3://{BUCKET_NAME}/raw/loans/',
            '--TARGET_PATH':      f's3://{BUCKET_NAME}/processed/loans_clean/',
            '--job-language':     'python',
            '--job-bookmark-option': 'job-bookmark-enable',
            '--enable-metrics':   '',
            '--enable-continuous-cloudwatch-log': 'true',
            '--enable-spark-ui':  'true',
            '--spark-event-logs-path': f's3://{BUCKET_NAME}/logs/glue/spark-ui/',
            '--TempDir':          f's3://{BUCKET_NAME}/logs/glue/temp/',
        },
        'GlueVersion':    '4.0',
        'WorkerType':     'G.1X',
        'NumberOfWorkers': 4,
        'Timeout':        120,  # minutes
        'MaxRetries':     1,
        'Tags': {
            'Project':     'credit-risk-early-warning',
            'Environment': 'dev',
            'Owner':       'dilip-chennam',
        }
    }

    try:
        glue.create_job(**job_config)
        print(f"  ✅ Job created: {JOB_NAME}")
    except glue.exceptions.AlreadyExistsException:
        glue.update_job(JobName=JOB_NAME, JobUpdate={
            k: v for k, v in job_config.items() if k != 'Name'
        })
        print(f"  ✅ Job updated: {JOB_NAME}")

def run_glue_job():
    """Trigger a job run and monitor it."""
    print(f"\nStarting Glue job run...")
    response = glue.start_job_run(JobName=JOB_NAME)
    run_id   = response['JobRunId']
    print(f"  Job Run ID: {run_id}")
    print(f"  Monitor at: https://{REGION}.console.aws.amazon.com/glue/home?region={REGION}#/etl/jobs/runs/{JOB_NAME}")

    print("\n  Monitoring job status (checking every 30s)...")
    while True:
        time.sleep(30)
        status = glue.get_job_run(JobName=JOB_NAME, RunId=run_id)
        state  = status['JobRun']['JobRunState']
        print(f"  Status: {state}")
        if state in ['SUCCEEDED', 'FAILED', 'ERROR', 'STOPPED']:
            break

    if state == 'SUCCEEDED':
        print(f"\n  ✅ Job completed successfully!")
    else:
        error = status['JobRun'].get('ErrorMessage', 'Unknown error')
        print(f"\n  ❌ Job failed: {error}")

    return run_id, state

def save_config(role_arn):
    config = {
        'job_name':   JOB_NAME,
        'role_name':  ROLE_NAME,
        'role_arn':   role_arn,
        'bucket':     BUCKET_NAME,
        'region':     REGION,
        'database':   'meridian_credit_risk',
        'source':     f's3://{BUCKET_NAME}/raw/loans/',
        'target':     f's3://{BUCKET_NAME}/processed/loans_clean/',
        'glue_console': f'https://{REGION}.console.aws.amazon.com/glue/home?region={REGION}'
    }
    with open('aws/glue/glue_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  ✅ Config saved to aws/glue/glue_config.json")

if __name__ == '__main__':
    print("=" * 60)
    print("Meridian Bank — Glue Job Provisioner")
    print("=" * 60)

    role_arn = create_glue_iam_role()
    create_glue_database()
    create_glue_job(role_arn)
    save_config(role_arn)

    print("\n" + "=" * 60)
    print("PROVISIONING COMPLETE")
    print("=" * 60)
    print(f"Job name:  {JOB_NAME}")
    print(f"Role:      {ROLE_NAME}")
    print(f"To run:    python aws/glue/provision_glue_job.py --run")
    print(f"Or via CLI: aws glue start-job-run --job-name {JOB_NAME}")
