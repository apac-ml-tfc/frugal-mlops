"""Lambda function to check current endpoint status and prepare for a (canary) deployment"""

# Python Built-Ins:
import json
import logging
import os

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
monitoring_bucket = os.environ["MONITORING_BUCKET"]

def handler(event, context):
    """Lambda handler to check current endpoint status and prepare configs for a (canary) deployment"""
    logger.info(event)

    endpoint_name = event["EndpointName"]
    target_model_name = event["ModelRegistration"]["Payload"]["ModelName"]
    result = {}

    # Check if the endpoint exists at all:
    try:
        endpoint_desc = smclient.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Found existing endpoint '{endpoint_name}'")
    except botoexceptions.ClientError as err:
        if err.response.get("Error", {}).get("Message", "").lower().startswith("could not find endpoint"):
            # Endpoint doesn't yet exist
            logger.info(f"Endpoint '{endpoint_name}' does not yet exist")
            endpoint_desc = None
        else:
            # Some other issue
            raise err

    # Check whether the endpoint already has multiple variants live (deployment in progress):
    if endpoint_desc is None:
        result["Status"] = "New"
    elif endpoint_desc["EndpointStatus"] in ("Creating", "Deleting"):  # Or could check != "InService"
        # TODO: Should this return a non-exception status like "Testing" does?
        raise RuntimeError(
            f"Endpoint '{existing_variant_name}' is currently in status '{endpoint_desc['EndpointStatus']}'"
        )
    elif len(endpoint_desc["ProductionVariants"]) > 1:
        # TODO: Any more diagnostic information needed?
        return { "Status": "Testing" }
    else:
        result["Status"] = "Ready"

    # OK Now we're ready to start creating our target (and maybe canary) configurations:
    # TODO: Add some parameter controls on data capture?
    data_capture_config = {
        "EnableCapture": True,
        "InitialSamplingPercentage": 50, # TODO: Parameterize & evolve?
        # A subfolder for endpoint name will automatically get created:
        "DestinationS3Uri": f"s3://{monitoring_bucket}/capture", 
        "CaptureOptions": [
            { "CaptureMode": "Input" },
            { "CaptureMode": "Output" },
        ],
    }

    # Target end-state variant for our new model, at 100% of traffic:
    target_variant_config = {
        "InitialInstanceCount": 1,  # TODO: Parameterize
        "InitialVariantWeight": 1.0,
        "InstanceType": "ml.g4dn.xlarge",  # TODO: Parameterize
        "ModelName": target_model_name,
        "VariantName": "blue",  # A starting assumption - we'll override below if needed
    }

    # End-state endpoint configuration:
    target_endpoint_config_name = util.append_timestamp(f"{endpoint_name}-target")
    target_config_create_response = smclient.create_endpoint_config(
        EndpointConfigName=target_endpoint_config_name,
        ProductionVariants=[target_variant_config],
        DataCaptureConfig=data_capture_config,
        Tags=[
            { "Key": "PipelineConfigType", "Value": "Target" },
        ],
    )
    result["TargetEndpointConfig"] = {
        "Arn": target_config_create_response["EndpointConfigArn"],
        "Name": target_endpoint_config_name,
    }

    # If an existing model is deployed, we'll also need to create an interim (canary monitoring) config:
    if endpoint_desc is not None:
        # Note there are slight differences between SageMaker API models for ProductionVariant (in
        # CreateEndpointConfiguration) and ProductionVariantSummary (from DescribeEndpoint):
        existing_variant_summary = endpoint_desc["ProductionVariants"][0]
        existing_variant_name = existing_variant_summary["VariantName"]
        if existing_variant_name == "blue":
            target_variant_config["VariantName"] = "green"
        existing_endpoint_config = smclient.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )
        try:
            # Variant names should be unique, so we'll cross-reference from the DescribeEndpoint:
            existing_variant_config = next(
                conf for conf in existing_endpoint_config["ProductionVariants"]
                if conf["VariantName"] == existing_variant_summary["VariantName"]
            )
        except StopIteration:
            # This could only happen if something else is manipulating the endpoint while this function runs,
            # but we should translate the cryptic error to what it actually means:
            raise RuntimeError("".join([
                f"Variant '{existing_variant_summary['VariantName']}' in endpoint '{existing_variant_name}'",
                f"DescribeEndpoint response was not present in follow-up DescribeEndpointConfiguration ",
                "result.",
            ]))

        # Now we've fetched the extra information, we can construct the new canary-period Variant configs for
        # the existing and new models:
        existing_variant_interim = json.loads(json.dumps(existing_variant_config))
        existing_variant_interim["InitialInstanceCount"] = existing_variant_summary["CurrentInstanceCount"]
        existing_variant_interim["InitialVariantWeight"] = 0.9  # TODO: Make canary pct configurable?
        new_variant_interim = json.loads(json.dumps(target_variant_config))
        new_variant_interim["InitialVariantWeight"] = 1. - existing_variant_interim["InitialVariantWeight"]

        interim_endpoint_config_name = util.append_timestamp(f"{endpoint_name}-canary")
        interim_config_create_response = smclient.create_endpoint_config(
            EndpointConfigName=interim_endpoint_config_name,
            ProductionVariants=[existing_variant_interim, new_variant_interim],
            DataCaptureConfig=data_capture_config,
            Tags=[
                { "Key": "PipelineConfigType", "Value": "Target" },
            ],
        )
        result["CanaryEndpointConfig"] = {
            "Arn": interim_config_create_response["EndpointConfigArn"],
            "Name": interim_endpoint_config_name,
        }

    # We've now prepped an endpoint config for target state, and for interim canary state if appropriate:
    logger.info(result)
    return result
