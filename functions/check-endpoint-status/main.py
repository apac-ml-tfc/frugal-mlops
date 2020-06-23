"""Lambda function to register a model from sandbox to project"""

# Python Built-Ins:
import logging

# External Dependencies:
import boto3
from botocore import exceptions as botoexceptions

# Fix logging in Lambda functions (before any local imports)
rootlogger = logging.getLogger()
if rootlogger.handlers:
    for handler in rootlogger.handlers:
        rootlogger.removeHandler(handler)
logging.basicConfig(level=logging.INFO)

# Local Dependencies:
import util


logger = logging.getLogger()

smclient = boto3.client("sagemaker")

def handler(event, context):
    logger.info(event)

    endpoint_name = event["EndpointName"]

    # Check if the endpoint exists at all:
    try:
        endpoint_desc = smclient.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Found existing endpoint '{endpoint_name}'")
    except botoexceptions.ClientError as err:
        if err.response.get("Error", {}).get("Message", "").lower().startswith("could not find endpoint"):
            # Endpoint doesn't yet exist
            logger.info(f"Endpoint '{endpoint_name}' does not yet exist")
            endpoint_desc = None
            return { "Status": "New" }
        else:
            # Some other issue
            raise err

    # Check whether the endpoint already has multiple variants live (deployment in progress):
    if len(endpoint_desc["ProductionVariants"]) > 1:
        # TODO: Some more diagnostic information?
        return { "Status": "Testing" }

    # Endpoint is active with exactly one production variant
    return {
        "Status": "Stable",
        "ActiveVariant": endpoint_desc["ProductionVariants"][0]
    }
