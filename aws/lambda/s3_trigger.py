"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
AWS Lambda: S3 Event Trigger for Glue ETL Job
================================================================================
This Lambda function is triggered by S3 PUT events on the raw/loans/ prefix.
When a new loan CSV lands in S3, it automatically starts the Glue ETL job.

Trigger configuration (set up via deploy_lambda.py):
  - Event source: S3 bucket — meridian-credit-risk-{account_id}
  - Event type:   s3:ObjectCreated:*
  - Prefix filter: raw/loans/

Author: Dilip Chennam
================================================================================
"""

import json
import boto3
import logging
import os
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

glue       = boto3.client('glue', region_name=os.environ.get('GLUE_REGION', 'us-west-2'))
JOB_NAME   = os.environ.get('GLUE_JOB_NAME', 'meridian-loan-data-transform')
SNS_TOPIC  = os.environ.get('SNS_TOPIC_ARN', '')

def lambda_handler(event, context):
    """
    Main Lambda handler. Processes S3 event, validates the trigger,
    and starts the Glue ETL job.
    """
    logger.info(f"Lambda triggered at {datetime.utcnow().isoformat()}")
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # ── Parse S3 event
        records = event.get('Records', [])
        if not records:
            logger.warning("No records in event — skipping")
            return {'statusCode': 200, 'body': 'No records'}

        triggered_files = []
        for record in records:
            bucket = record['s3']['bucket']['name']
            key    = record['s3']['object']['key']
            size   = record['s3']['object'].get('size', 0)
            logger.info(f"New file detected: s3://{bucket}/{key} ({size:,} bytes)")
            triggered_files.append({'bucket': bucket, 'key': key, 'size': size})

        # ── Validate file type
        csv_files = [f for f in triggered_files if f['key'].endswith('.csv')]
        if not csv_files:
            logger.info("No CSV files in event — skipping Glue trigger")
            return {'statusCode': 200, 'body': 'No CSV files'}

        # ── Check if Glue job is already running
        running_jobs = glue.get_job_runs(JobName=JOB_NAME, MaxResults=5)
        active_runs  = [
            r for r in running_jobs.get('JobRuns', [])
            if r['JobRunState'] in ['RUNNING', 'STARTING', 'STOPPING']
        ]
        if active_runs:
            run_id = active_runs[0]['Id']
            logger.warning(f"Glue job already running (Run ID: {run_id}) — skipping duplicate trigger")
            return {
                'statusCode': 200,
                'body': f'Job already running: {run_id}'
            }

        # ── Start Glue job
        logger.info(f"Starting Glue job: {JOB_NAME}")
        response = glue.start_job_run(
            JobName=JOB_NAME,
            Arguments={
                '--trigger_source': 'lambda_s3_event',
                '--triggered_files': json.dumps([f['key'] for f in csv_files]),
                '--triggered_at':    datetime.utcnow().isoformat(),
            }
        )
        run_id = response['JobRunId']
        logger.info(f"  ✅ Glue job started. Run ID: {run_id}")

        # ── Optional: send SNS notification
        if SNS_TOPIC:
            sns = boto3.client('sns')
            sns.publish(
                TopicArn=SNS_TOPIC,
                Subject=f'Meridian Bank ETL Job Started',
                Message=json.dumps({
                    'job_name':       JOB_NAME,
                    'run_id':         run_id,
                    'triggered_by':   'S3 event',
                    'files':          [f['key'] for f in csv_files],
                    'triggered_at':   datetime.utcnow().isoformat(),
                }, indent=2)
            )
            logger.info(f"  ✅ SNS notification sent")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message':    f'Glue job started successfully',
                'job_name':   JOB_NAME,
                'run_id':     run_id,
                'files':      [f['key'] for f in csv_files],
            })
        }

    except glue.exceptions.ConcurrentRunsExceededException:
        logger.warning("Glue job concurrent run limit exceeded — will retry on next event")
        return {'statusCode': 429, 'body': 'Concurrent run limit exceeded'}

    except Exception as e:
        logger.error(f"Lambda error: {str(e)}", exc_info=True)
        raise
