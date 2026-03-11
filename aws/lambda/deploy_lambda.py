"""
================================================================================
Meridian Bank — Credit Risk Early Warning System
Lambda Deployment Script
================================================================================
Packages and deploys the S3 trigger Lambda function to AWS.
Run AFTER setup_s3.py and provision_glue_job.py.

Author: Dilip Chennam
================================================================================
"""

import boto3
import json
import zipfile
import io
import time
import os

REGION        = 'us-west-2'
FUNCTION_NAME = 'meridian-s3-glue-trigger'
ROLE_NAME     = 'MeridianLambdaExecutionRole'
JOB_NAME      = 'meridian-loan-data-transform'

sts  = boto3.client('sts',    region_name=REGION)
lam  = boto3.client('lambda', region_name=REGION)
iam  = boto3.client('iam',    region_name=REGION)
s3   = boto3.client('s3',     region_name=REGION)

ACCOUNT_ID  = sts.get_caller_identity()['Account']
BUCKET_NAME = f'meridian-credit-risk-{ACCOUNT_ID}'

def create_lambda_role():
    print(f"Creating Lambda IAM role: {ROLE_NAME}")
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    try:
        r = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description='Lambda execution role for Meridian S3 trigger'
        )
        role_arn = r['Role']['Arn']
        print(f"  ✅ Role created: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
        print(f"  ℹ️  Role exists: {role_arn}")

    for policy in [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/AWSGlueConsoleFullAccess',
    ]:
        try:
            iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy)
        except: pass

    time.sleep(10)
    return role_arn

def package_lambda():
    print("Packaging Lambda function...")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write('aws/lambda/s3_trigger.py', 'lambda_function.py')
    buffer.seek(0)
    print("  ✅ Packaged lambda_function.py")
    return buffer.read()

def deploy_lambda(role_arn, zip_bytes):
    print(f"\nDeploying Lambda: {FUNCTION_NAME}")
    env = {
        'GLUE_JOB_NAME': JOB_NAME,
        
    }
    try:
        lam.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime='python3.11',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_bytes},
            Description='Triggers Glue ETL when new loan CSV lands in S3',
            Timeout=60,
            MemorySize=256,
            Environment={'Variables': env},
            Tags={
                'Project': 'credit-risk-early-warning',
                'Owner':   'dilip-chennam',
            }
        )
        print(f"  ✅ Function created: {FUNCTION_NAME}")
    except lam.exceptions.ResourceConflictException:
        lam.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
        lam.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Environment={'Variables': env}
        )
        print(f"  ✅ Function updated: {FUNCTION_NAME}")

def add_s3_trigger():
    print(f"\nAdding S3 event trigger...")
    func_arn = lam.get_function(FunctionName=FUNCTION_NAME)['Configuration']['FunctionArn']

    # Allow S3 to invoke Lambda
    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId='S3InvokePermission',
            Action='lambda:InvokeFunction',
            Principal='s3.amazonaws.com',
            SourceArn=f'arn:aws:s3:::{BUCKET_NAME}',
            SourceAccount=ACCOUNT_ID,
        )
        print(f"  ✅ S3 invoke permission granted")
    except lam.exceptions.ResourceConflictException:
        print(f"  ℹ️  Permission already exists")

    # Set S3 bucket notification
    s3.put_bucket_notification_configuration(
        Bucket=BUCKET_NAME,
        NotificationConfiguration={
            'LambdaFunctionConfigurations': [{
                'LambdaFunctionArn': func_arn,
                'Events': ['s3:ObjectCreated:*'],
                'Filter': {
                    'Key': {
                        'FilterRules': [
                            {'Name': 'prefix', 'Value': 'raw/loans/'},
                            {'Name': 'suffix', 'Value': '.csv'}
                        ]
                    }
                }
            }]
        }
    )
    print(f"  ✅ S3 trigger configured: raw/loans/*.csv → {FUNCTION_NAME}")

if __name__ == '__main__':
    print("=" * 60)
    print("Meridian Bank — Lambda Deployment")
    print("=" * 60)
    role_arn  = create_lambda_role()
    zip_bytes = package_lambda()
    deploy_lambda(role_arn, zip_bytes)
    add_s3_trigger()
    print("\n✅ Lambda deployed and S3 trigger active")
    print(f"Any new CSV in s3://{BUCKET_NAME}/raw/loans/ will trigger the Glue ETL job automatically")
