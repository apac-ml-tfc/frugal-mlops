"""Custom CloudFormation Resource to describe details of an existing SMStudio user profile

For details of published CloudFormation resource output properties, see docstring for the
user_description_to_cfn_result() function.
"""

# Python Built-Ins:
import logging
import time
import traceback

# External Dependencies:
import boto3
import cfnresponse

logger = logging.getLogger("main")
smclient = boto3.client("sagemaker")

# The set of DescribeDomain response props that will be passed through to cfn output:
PASSTHRU_PROPS = {
    "DomainId",
    "UserProfileArn",
    "UserProfileName",
    "HomeEfsFileSystemUid",
    "Status",
    "SingleSignOnUserIdentifier",
    "SingleSignOnUserValue",
    "AppNetworkAccessType",
}

def lambda_handler(event, context):
    try:
        request_type = event["RequestType"]
        if request_type == "Create":
            logger.info("**Received create event")
            handle_create_or_update(event, context)
        elif request_type == "Update":
            logger.info("**Received update event")
            handle_create_or_update(event, context)
        elif request_type == "Delete":
            logger.info("**Received delete event")
            handle_delete(event, context)
        else:
            cfnresponse.send(
                event,
                context,
                cfnresponse.FAILED,
                {},
                error=f"Unsupported CFN RequestType '{request_type}'",
            )
    except Exception as e:
        logger.error("Uncaught exception in CFN custom resource handler - reporting failure")
        traceback.print_exc()
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {},
            error=str(e),
        )
        raise e


class NoStudioDomains(RuntimeError):
    pass


def infer_domain_id():
    domains_resp = smclient.list_domains()
    if "NextToken" in domains_resp:
        logger.warning(
            f"Ignoring NextToken on sagemaker:ListDomains response - pagination not implemented"
        )
    domain_ids = [d["DomainId"] for d in domains_resp["Domains"]]

    if not (len(domain_ids) > 0):
        # If the domain has been deleted, the user must necessarily have been deleted too!
        raise NoStudioDomains(f"No SageMaker Studio domain exists in this region!")
    elif len(domain_ids) > 1:
        logger.warning(
            f"Found {len(domain_ids)} Studio domains in this region: assuming first is target. {domain_ids}"
        )
    return domain_ids[0]


def describe_stable_user(domain_id, user_profile_name, poll_secs=30, max_wait_secs=60*20):
    """Poll SageMaker DescribeUserProfile until the user is in a stable status (not updating, etc)
    """
    updating_statuses = { "Deleting", "Pending", "Updating" }
    last_status = "Pending"
    t0 = time.time()

    while (time.time() - t0) < max_wait_secs:
        desc = smclient.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
        last_status = desc["Status"]
        if last_status not in updating_statuses:
            return desc
        time.sleep(poll_secs)
    raise RuntimeError(
        f"Timed out after {time.time() - t0} seconds before user stabilized! Last status: {last_status}"
    )


def flatten_nested_dict(obj, parent_key=""):
    """Flatten the keys of any nested dicts in 'obj' with dot notation, and exclude non-top-level lists"""
    result = {}
    for rawkey, val in obj.items():
        if parent_key and isinstance(val, list):
            # We don't currently support lists nested in the profile (i.e. custom kernel list)
            continue
        key = f"{parent_key}.{rawkey}" if parent_key else rawkey
        result[key] = flatten_nested_dict(val, parent_key=key) if isinstance(val, dict) else val
    return result


def user_description_to_cfn_result(desc):
    """Derive CloudFormation resource outputs from a SageMaker User Profile description

    CloudFormation resource outputs don't support nested objects, so we derive the result as follows:

    - Properties listed in `PASSTHRU_PROPS` constant in this module are passed through as-is
    - The contents of the `UserSettings` object are mapped to top-level properties (since there are no
      conflicts and it's more concise)
    - Nested dict contents of `UserSettings` are flattened with dot notation, so they can be naturally
      accessed via CloudFormation: E.g. !GetAtt MyExistingUP.SharingSettings.S3OutputPath
    - Non-top-level lists in `UserSettings` (i.e. KernelGatewayAppSettings.CustomImages` are dropped as
      there's no nice way to access them.
    """
    result = { k: desc[k] for k in desc.keys() if k in PASSTHRU_PROPS }
    user_settings_flat = flatten_nested_dict(desc["UserSettings"])
    for k, v in user_settings_flat.items():
        result[k] = v
    user_exec_role = user_settings_flat.get("ExecutionRole")
    if user_exec_role:  # Just in case it's not set for some reason?
        result["ExecutionRoleName"] = user_exec_role.rpartition("/")[2]  # Name from ARN
    return result


def handle_create_or_update(event, context):
    """Update is completely the same process as create for now, as it's just a descriptive resource"""
    user_profile_name = event["ResourceProperties"]["UserProfileName"]
    domain_id = event["ResourceProperties"].get("DomainId")

    if domain_id is None:
        logger.info("Inferring domain ID")
        domain_id = infer_domain_id()

    logger.info(f"Querying user {user_profile_name} on domain {domain_id}")
    desc = describe_stable_user(domain_id, user_profile_name)
    result = user_description_to_cfn_result(desc)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        result,
        physicalResourceId=desc["UserProfileArn"],
    )


def handle_delete(event, context):
    """Descriptive resource has nothing to actually delete, but needs to report success to CloudFormation"""
    logger.info("Descriptive resource - nothing to delete")
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )
