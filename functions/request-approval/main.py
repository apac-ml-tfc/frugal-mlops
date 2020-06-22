"""Lambda to send approval emails from Step Functions events"""

# Python Built-Ins:
import json
import logging
import os
from string import Template
import urllib

# External Dependencies:
import boto3

# Fix logging in Lambda functions (before any local imports)
rootlogger = logging.getLogger()
if rootlogger.handlers:
    for handler in rootlogger.handlers:
        rootlogger.removeHandler(handler)
logging.basicConfig(level=logging.INFO)

# Local Dependencies:
import util


logger = logging.getLogger()

botosess = boto3.session.Session()
region_name = botosess.region_name
ses = botosess.client("ses")
sfn = botosess.client("stepfunctions")
sns = botosess.client("sns")

with open("email.tpl.html", "r") as f:
    email_template = Template(f.read())


def handler(event, context):
    """Lambda to send approval emails from Step Functions events"""
    logger.info(f"Got event {event}")

    execution_context = event["ExecutionContext"]

    # (This part is only really for generating a nice AWS Console URL)
    execution_name = execution_context["Execution"]["Name"]
    lambda_arn_tokens = context.invoked_function_arn.split(":")
    partition = lambda_arn_tokens[1]
    account_id = lambda_arn_tokens[4]
    state_machine_name = execution_context["StateMachine"]["Name"]
    execution_arn = "arn:{}:states:{}:{}:execution:{}:{}".format(
        partition,
        region_name,
        account_id,
        state_machine_name,
        execution_name,
    )
    details_url = f"https://console.aws.amazon.com/states/home?region={region_name}#/executions/details/{execution_arn}"

    # Now for the real task processing:
    task_token = execution_context["Task"]["Token"]
    task_token_uriencoded = urllib.parse.quote(task_token)
    approval_uri = event["ApprovalUri"]
    approval_uri = "".join([
        approval_uri,
        # (Base approval URI from input may or may not have other query params)
        "&" if "?" in approval_uri else "?",
        "taskToken=",
        task_token_uriencoded,
    ])
    rejection_uri = event["RejectionUri"]
    rejection_uri = "".join([
        rejection_uri,
        # (Base rejection URI from input may or may not have other query params)
        "&" if "?" in rejection_uri else "?",
        "taskToken=",
        task_token_uriencoded,
    ])

    # Must have *either* an email address (for SES) or a topic (SNS)
    manager_email = event.get("ManagerEmailAddress")
    sns_topic = event.get("EmailTopic")
    timeout_description = event["TimeoutDescription"]

    if manager_email:
        # If an email address is provided, try it first because we can send richer (HTML) content:
        try:
            no_reply_email = "no-reply@" + manager_email.partition("@")[2]  # At same domain
            logger.info("Sending email...")
            ses.send_email(
                Source=manager_email,  # Tag as from self
                ReplyToAddresses=[no_reply_email],
                Destination={ "ToAddresses": [manager_email] },
                Message={
                    "Subject": {
                        "Charset": "UTF-8",
                        "Data": "Your approval needed for model deployment",
                    },
                    "Body": {
                        "Html": {
                            "Charset": "UTF-8",
                            "Data": email_template.safe_substitute({
                                "ApproveLink": approval_uri,
                                "RejectLink": rejection_uri,
                                "ModelName": "TODO",
                                "ModelScore": "TODO",
                                "Timeout": timeout_description,
                                "DetailsUrl": details_url,
                            }),
                        },
                    },
                },
            )
            logger.info("Email sent")
            return
        except Exception as e:
            if sns_topic:
                # Fall back to SNS topic
                logger.warning("Failed to send email: Defaulting to provided SNS topic", exc_info=True)
            else:
                # Failed to send email and no backup option available
                raise e

    # Fallback option: SNS notification
    sns_response = sns.publish(
        TopicArn=sns_topic,
        Subject="Your approval needed for model deployment",
        Message="".join([
            "Hello,\n\n",
            "A new model has been tested and is ready for deployment.\n\n",
            # TODO: Model name and score
            "Please *approve* to trigger phased deployment, or *reject* the change within ",
            f"{timeout_description}, or the model will be auto-rejected.\n\n\n",
            f"Approve -> {approval_uri}\n\n",
            f"Reject -> {rejection_uri}\n\n\n\n",
            f"To view the current status of this workflow in the AWS Console, visit: {details_url}\n\n"
        ])
    )
    logger.info(f"SNS notification sent {sns_response}")
