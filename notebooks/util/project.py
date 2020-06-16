"""ML Project Infrastructure utilities"""

# Python Built-Ins:
from datetime import datetime, date
import json

# External Dependencies:
import boto3

# Local Dependencies:
from .uid import append_timestamp

sfn = boto3.client("stepfunctions")

def stringify_datetime(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not JSON serializable" % type(obj))


def submit_model(data):
    """Submit a candidate model from data science workbench to project approval flow

    Simply starts a Step Function for the model approval/deployment pipeline, and returns the result
    """
    # Check recorded against a Trial
    # Perform a validation batch transform
    
    return sfn.start_execution(
        stateMachineArn=,  # TODO: Pass through from ML project setup
        name=append_timestamp("execution"),  # Auto-derive something so we can make use of re-use checking
        input=json.dumps(data, default=stringify_datetime),
    )

def submission_status(submission):
    """Check the status of an ML model submission (== a Step Function execution)"""
    execution_arn = submission if isinstance(submission, str) else submission["executionArn"]
    return sfn.describe_execution(executionArn=execution_arn)
