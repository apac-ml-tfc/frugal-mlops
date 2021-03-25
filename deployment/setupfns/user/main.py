"""Custom CloudFormation Resource for a SageMaker Studio User Profile"""

# Python Built-Ins:
import json
import logging
import time
import traceback

# External Dependencies:
import boto3
from botocore.exceptions import ClientError
import cfnresponse

smclient = boto3.client("sagemaker")


def lambda_handler(event, context):
    try:
        request_type = event["RequestType"]
        if request_type == "Create":
            handle_create(event, context)
        elif request_type == "Update":
            handle_update(event, context)
        elif request_type == "Delete":
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
        logging.error("Uncaught exception in CFN custom resource handler - reporting failure")
        traceback.print_exc()
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {},
            error=str(e),
        )
        raise e


def handle_create(event, context):
    logging.info("**Received create request")
    resource_config = event["ResourceProperties"]

    logging.info("**Creating user profile")
    result = create_user_profile(resource_config)
    # TODO: Do we need to wait for completion?
    response = {
        "UserProfileName": result["UserProfileName"],
        "HomeEfsFileSystemUid": result["HomeEfsFileSystemUid"],
    }
    print(response)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        response,
        physicalResourceId=result["UserProfileName"],
    )


def handle_delete(event, context):
    logging.info("**Received delete event")
    user_profile_name = event["PhysicalResourceId"]
    domain_id = event["ResourceProperties"].get("DomainId")

    if domain_id is None:
        try:
            domains_resp = smclient.list_domains()
            if "NextToken" in domains_resp:
                logging.warning(
                    f"Ignoring NextToken on sagemaker:ListDomains response - pagination not implemented"
                )
            domain_ids = [d["DomainId"] for d in domains_resp["Domains"]]
        except Exception as e:
            raise ValueError(
                "DomainId not explicitly provided and could not query sagemaker:ListDomains"
            ) from e
        if not (len(domain_ids) > 0):
            # If the domain has been deleted, the user must necessarily have been deleted too!
            logging.info(f"Domain already deleted - skipping user deletion")
            cfnresponse.send(
                event,
                context,
                cfnresponse.SUCCESS,
                {},
                physicalResourceId=event["PhysicalResourceId"],
            )
            return
        domain_id = domain_ids[0]

    try:
        smclient.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
    except smclient.exceptions.ResourceNotFound as exception:
        cfnresponse.send(
            event,
            context,
            cfnresponse.SUCCESS,
            {},
            physicalResourceId=event["PhysicalResourceId"],
        )
        return
    delete_user_profile(domain_id, user_profile_name)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def handle_update(event, context):
    logging.info("**Received update event")
    user_profile_name = event["PhysicalResourceId"]
    domain_id = event["ResourceProperties"].get("DomainId")

    if domain_id is None:
        try:
            domains_resp = smclient.list_domains()
            if "NextToken" in domains_resp:
                logging.warning(
                    f"Ignoring NextToken on sagemaker:ListDomains response - pagination not implemented"
                )
            domain_ids = [d["DomainId"] for d in domains_resp["Domains"]]
        except Exception as e:
            raise ValueError(
                "DomainId not explicitly provided and could not query sagemaker:ListDomains"
            ) from e
        if not (len(domain_ids) > 0):
            # If the domain has been deleted, the user must necessarily have been deleted too!
            raise ValueError(f"No SageMaker Studio domain available in this region!")
        domain_id = domain_ids[0]

    user_settings = event["ResourceProperties"]["UserSettings"]
    update_user_profile(domain_id, user_profile_name, user_settings)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def create_user_profile(config):
    domain_id = config.get("DomainId")

    if domain_id is None:
        try:
            domains_resp = smclient.list_domains()
            if "NextToken" in domains_resp:
                logging.warning(
                    f"Ignoring NextToken on sagemaker:ListDomains response - pagination not implemented"
                )
            domain_ids = [d["DomainId"] for d in domains_resp["Domains"]]
        except Exception as e:
            raise ValueError(
                "DomainId not explicitly provided and could not query sagemaker:ListDomains"
            ) from e
        if not (len(domain_ids) > 0):
            # If the domain has been deleted, the user must necessarily have been deleted too!
            logging.info(f"Domain already deleted - skipping user deletion")
            raise ValueError(f"No SageMaker Studio domain available in this region!")
        domain_id = domain_ids[0]

    user_profile_name = config["UserProfileName"]
    user_settings = config["UserSettings"]

    response = smclient.create_user_profile(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        UserSettings=user_settings,
    )
    created = False
    time.sleep(0.2)
    while not created:
        response = smclient.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
        status_lower = response["Status"].lower()
        if status_lower == "inservice":
            created = True
            break
        elif "failed" in status_lower:
            raise ValueError(
                f"User '{user_profile_name}' entered Failed state during creation (domain {domain_id})",
            )
        time.sleep(5)

    logging.info("**SageMaker domain created successfully: %s", domain_id)
    return response


def delete_user_profile(domain_id, user_profile_name):
    response = smclient.delete_user_profile(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
    )
    deleted = False
    time.sleep(0.2)
    while not deleted:
        try:
            response = smclient.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
            status_lower = response["Status"].lower()
            if "failed" in status_lower:
                raise ValueError(
                    f"User '{user_profile_name}' entered Failed state during deletion (domain {domain_id})",
                )
            elif "deleting" not in status_lower:
                raise ValueError(
                    f"User '{user_profile_name}' no longer 'Deleting' but not deleted (domain {domain_id})",
                )
        except smclient.exceptions.ResourceNotFound:
            logging.info("Deleted user %s from domain %s", user_profile_name, domain_id)
            deleted = True
            break
        time.sleep(5)
    return response


def update_user_profile(domain_id, user_profile_name, user_settings):
    response = smclient.update_user_profile(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        UserSettings=user_settings,
    )
    updated = False
    time.sleep(0.2)
    while not updated:
        response = smclient.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
        status_lower = response["Status"].lower()
        if status_lower == "inservice":
            updated = True
            break
        elif "failed" in status_lower:
            raise ValueError(
                f"User '{user_profile_name}' entered Failed state during deletion (domain {domain_id})",
            )
        time.sleep(5)
    return response
