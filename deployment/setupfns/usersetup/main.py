"""Custom CloudFormation Resource for post-creation setup of a SageMaker Studio user

Functionality includes:
- Cloning a (public) 'GitRepository' into the user's home folder (assuming this Lambda has mounted the
  SMStudio EFS filesystem as root and has access to the repository)
- Enabling SageMaker Project templates functionality for the user

Updating or deleting this resource does not currently do anything. Errors in the content copy process are
also ignored (typically don't want to roll back the whole stack just because we couldn't clone a repo - as
users can always do it manually!)
"""

# Python Built-Ins:
import json
import logging
import os
import time
import traceback

# External Dependencies:
import boto3
from botocore.exceptions import ClientError
import cfnresponse

# Local Dependencies:
import content
import smprojects

logger = logging.getLogger("main")
smclient = boto3.client("sagemaker")


def lambda_handler(event, context):
    try:
        request_type = event["RequestType"]
        if request_type == "Create":
            logger.info("**Received create event")
            handle_create(event, context)
        elif request_type == "Update":
            logger.info("**Received update event")
            handle_update(event, context)
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


def handle_create(event, context):
    resource_config = event["ResourceProperties"]
    logger.info("**Setting up user")
    result = create_user_setup(resource_config)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        { "UserProfileName": result["UserProfileName"] },
        physicalResourceId=result["UserProfileName"],
    )


def handle_delete(event, context):
    user_profile_name = event["PhysicalResourceId"]
    domain_id = event["ResourceProperties"]["DomainId"]
    enable_projects = event["ResourceProperties"].get("EnableProjects", False)
    logger.info("**Deleting user setup")
    delete_user_setup(domain_id, user_profile_name, enable_projects=enable_projects)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def handle_update(event, context):
    user_profile_name = event["PhysicalResourceId"]
    domain_id = event["ResourceProperties"]["DomainId"]
    git_repo = event["ResourceProperties"]["GitRepository"]
    logger.info("**Updating user setup")
    update_user_setup(domain_id, user_profile_name, git_repo)
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def create_user_setup(config):
    domain_id = config["DomainId"]
    user_profile_name = config["UserProfileName"]
    git_repo = config.get("GitRepository")
    efs_uid = config.get("HomeEfsFileSystemUid")
    enable_projects = config.get("EnableProjects", False)

    print(f"Setting up user: {config}")
    # Clone in the GitRepository, if requested:
    if git_repo:
        if not efs_uid: raise ValueError(
            "HomeEfsFileSystemUid parameter is mandatory when GitRepository is specified"
        )
        try:
            content.clone_git_repository(efs_uid, git_repo)
        except Exception as e:
            # Don't bring the entire CF stack down just because we couldn't copy a repo:
            print("IGNORING CONTENT SETUP ERROR")
            traceback.print_exc()

    ## Enable SageMaker Projects/JumpStart if requested:
    if enable_projects:
        # We need to look up the role ARN for the user:
        user_desc = smclient.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
        user_role_arn = user_desc["UserSettings"]["ExecutionRole"]
        smprojects.enable_sm_projects_for_role(user_role_arn)

    logger.info("**SageMaker Studio user '%s' set up successfully", user_profile_name)
    return { "UserProfileName": user_profile_name }


def delete_user_setup(domain_id, user_profile_name, enable_projects=False):
    logger.info(
        "**Deleting user setup is a no-op: user '%s' on domain '%s",
        user_profile_name,
        domain_id,
    )

    ## Disable SageMaker Projects/JumpStart if requested:
    if enable_projects:
        # We need to look up the role ARN for the user:
        user_desc = smclient.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
        user_role_arn = user_desc["UserSettings"]["ExecutionRole"]
        smprojects.disable_sm_projects_for_role(user_role_arn)

    return { "UserProfileName": user_profile_name }


def update_user_setup(domain_id, user_profile_name, git_repo):
    logger.warning(
        "**Updating user setup is a no-op: user '%s' on domain '%s",
        user_profile_name,
        domain_id,
    )
    return { "UserProfileName": user_profile_name }
