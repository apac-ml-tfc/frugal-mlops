"""ML Project Infrastructure utilities

init() with a valid *project ID* (or provide the PROJECT_ID environment variable), and this module will read
the project's configuration from AWS SSM (stored there by the CloudFormation stack): Allowing us to provide
project-specific utility functions e.g. submitting a model for release, etc.

A "session" in the context of this module is a project config loaded from SSM. This way we can choose either
to init-and-forget (standard data science project sandbox use case) or to call individual functions on
separate sessions (interact with multiple projects).
"""

# Python Built-Ins:
from datetime import datetime, date
import json
import logging
import os
from types import SimpleNamespace

# External Dependencies:
import boto3

# Local Dependencies:
from .uid import append_timestamp
from . import progress

logger = logging.getLogger("project")

sfn = boto3.client("stepfunctions")
ssm = boto3.client("ssm")


defaults = SimpleNamespace()
defaults.project_id = None
defaults.session = None

if "PROJECT_ID" not in os.environ:
    logger.info("No PROJECT_ID variable found in environment: You'll need to call init('myprojectid')")
else:
    defaults.project_id = os.environ["PROJECT_ID"]


def get_session(project_id: str):
    """Get a new session/configuration object for a ML Project ID"""
    param_ids = [
        "ArtifactsBucket",
        "CodeCommit",
        "MonitoringBucket",
        "PipelineStateMachine",
        "SourceBucket",
        "SudoRole"
    ]
    param_ssm_names = list(map(lambda s: f"/{project_id}-Project/{s}", param_ids))
    ssm_param_id_map = { param_ssm_names[i]: param_ids[i] for i in range(len(param_ids)) }

    response = ssm.get_parameters(Names=param_ssm_names)
    n_invalid = len(response.get("InvalidParameters", []))
    if n_invalid == param_ids:
        raise ValueError(f"Found no valid SSM parameters for /{project_id}-Project: Invalid project ID")
    elif n_invalid > 0:
        logger.warning(" ".join([
            f"{n_invalid} Project parameters missing from SSM: Some functionality may not work as expected.",
            f"Missing: {response['InvalidParameters']}"
        ]))

    result = { "ProjectId": project_id }
    for param in response["Parameters"]:
        result[ssm_param_id_map[param["Name"]]] = param["Value"]
    return result


def init(project_id):
    """Initialise the project util library (and the default session) to project_id"""
    # Check that we can create the session straight away, for nice error behaviour:
    session = get_session(project_id)
    if defaults.project_id and defaults.project_id != project_id and defaults.session:
        logger.info(f"Clearing previous default session for project '{defaults.project_id}'")
    defaults.project_id = project_id
    defaults.session = session
    logger.info(f"Working in project '{project_id}'")
    return session


def session_or_default(sess=None):
    """Mostly-internal utility to return either the provided session or else a default"""
    if sess:
        return sess
    elif defaults.session:
        return defaults.session
    elif defaults.project_id:
        defaults.session = get_session(defaults.project_id)
        return defaults.session
    else:
        raise ValueError(
            "Must provide a project session or init() the project library with a valid project ID"
        )


def stringify_datetime(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not JSON serializable" % type(obj))


def submit_model(data, session=None, wait=False):
    """Submit a candidate model from data science workbench to project approval flow

    If wait is falsy, simply starts a Step Function for the model approval/deployment pipeline, and returns
    the response. Otherwise, displays a spinner and waits for the submission workflow to complete.
    """
    # TODO: Check recorded against a Trial
    submission = sfn.start_execution(
        stateMachineArn=session_or_default(session)["PipelineStateMachine"],
        name=append_timestamp("execution"),  # Auto-derive something so we can make use of re-use checking
        input=json.dumps(data, default=stringify_datetime),
    )
    if not wait:
        return submission

    # Else wait with a spinner:
    def is_registered(status):
        statstr = status["status"]
        if statstr == "SUCCEEDED":
            return True
        elif statstr == "RUNNING":
            return False
        else:
            raise ValueError(f"Stopped with non-success status {statstr}")
    return progress.sfn_polling_spinner(
        submission["executionArn"],
        poll_secs = 5,
    )
