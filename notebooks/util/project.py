"""ML Project Infrastructure utilities

init() with a valid *project ID* (or provide the PROJECT_ID environment variable), and this module will read
the project's configuration from AWS SSM (stored there by the CloudFormation stack): Allowing us to provide
project-specific utility functions e.g. submitting a model for release, etc.

A "session" in the context of this module is a project config loaded from SSM. This way we can choose either
to init-and-forget (standard data science project sandbox use case) or to call individual functions on
separate sessions (interact with multiple projects).
"""

# Python Built-Ins:
from collections import namedtuple
from datetime import datetime, date
import json
import logging
import os
from types import SimpleNamespace
from typing import Union

# External Dependencies:
import boto3
import sagemaker

# Local Dependencies:
from . import uid
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


class ProjectSession:
    # This is implemented as a class to give a nice documented Python API of what should be present, but that
    # has made the code a little verbose for fetching SSM params and setting each one into the correct prop.
    def __init__(self, project_id: str, role: Union[str, None]=None):
        """Session/configuration object for an ML project and the current user's sandbox environment

        Parameters
        ----------
        project_id :
            ID of the provisioned ML project stack
        role : (optional)
            ARN of the current user's role (can use sagemaker.get_execution_role() if not sure). Include to
            also fetch individual sandbox parameters (e.g. sandbox bucket name). Exclude to fetch only shared
            project-level parameters (e.g. shared artifacts bucket name).
        """
        self.project_id = project_id
        self.role = role
        shared_param_ids = [
            "ArtifactsBucket",
            "CodeCommit",
            "MonitoringBucket",
            "PipelineStateMachine",
            "SourceBucket",
            "SudoRole",
        ]
        self.artifacts_bucket = None
        self.code_commit = None
        self.monitoring_bucket = None
        self.pipeline_state_machine = None
        self.source_bucket = None
        self.sudo_role = None
        param_ssm_names = {
            f"/{project_id}-Project/{s}": { "cat": "shared", "id": s } for s in shared_param_ids
        }

        if role is None:
            self.sandbox = None
        else:
            role_name = role.partition("/")[2] if role.startswith("arn:") else role
            sandbox_param_ids = ["ArtifactsBucket", "SandboxBucket"]
            self.sandbox = SimpleNamespace(artifacts_bucket=None, sandbox_bucket=None)
            for s in sandbox_param_ids:
                param_ssm_names[f"/{project_id}-Project/{role_name}/{s}"] = { "cat": "sandbox", "id": s }

        response = ssm.get_parameters(Names=[s for s in param_ssm_names])
        n_invalid = len(response.get("InvalidParameters", []))
        if n_invalid == len(param_ssm_names):
            raise ValueError(f"Found no valid SSM parameters for /{project_id}-Project: Invalid project ID")
        elif n_invalid > 0:
            logger.warning(" ".join([
                f"{n_invalid} Project parameters missing from SSM: Some functionality may not work as",
                f"expected. Missing: {response['InvalidParameters']}"
            ]))

        for param in response["Parameters"]:
            param_spec = param_ssm_names[param["Name"]]
            param_id = param_spec["id"]
            if param_spec["cat"] == "shared":
                if param_id == "ArtifactsBucket":
                    self.artifacts_bucket = param["Value"]
                elif param_id == "CodeCommit":
                    self.code_commit = param["Value"]
                elif param_id == "MonitoringBucket":
                    self.monitoring_bucket = param["Value"]
                elif param_id == "PipelineStateMachine":
                    self.pipeline_state_machine = param["Value"]
                elif param_id == "SourceBucket":
                    self.source_bucket = param["Value"]
                elif param_id == "SudoRole":
                    self.sudo_role = param["Value"]
                else:
                    raise ValueError(
                        "Mistake in util ProjectSession implementation. Got unexpected SSM param {}".format(
                            param_id
                        )
                    )
            elif param_spec["cat"] == "sandbox":
                if param_id == "ArtifactsBucket":
                    self.sandbox.artifacts_bucket = param["Value"]
                elif param_id == "SandboxBucket":
                    self.sandbox.sandbox_bucket = param["Value"]
                else:
                    raise ValueError(
                        "Mistake in util ProjectSession implementation. Got unexpected SSM param {}".format(
                            param_id
                        )
                    )
            else:
                raise ValueError(
                    "Mistake in util ProjectSession implementation. Got unexpected SSM param type {}".format(
                        param_spec
                    )
                )

    def submit_model(self, data, wait=False):
        """Submit a candidate model from data science workbench to project approval flow

        If wait is falsy, simply start a Step Function for the model approval/deployment pipeline, and return
        the response. Otherwise, display a spinner and waits for the submission workflow to complete.
        """
        # TODO: Check recorded against a Trial
        submission = sfn.start_execution(
            stateMachineArn=self.pipeline_state_machine,
            name=uid.append_timestamp("execution"),  # Auto-derive so we can make use of re-use checking
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

    def __repr__(self):
        """Strike a balance when this class is print()ed between similarity to standard repr, and readability
        """
        typ = type(self)
        mod = typ.__module__
        qualname = typ.__qualname__
        propdict = self.__dict__
        proprepr = ",\n  ".join([f"{k}={propdict[k]}" for k in propdict])
        return f"<{mod}.{qualname}(\n  {proprepr}\n) at {hex(id(self))}>"


def init(project_id: str, role: Union[str, None]=None) -> ProjectSession:
    """Initialise the project util library (and the default session) to project_id"""
    # Check that we can create the session straight away, for nice error behaviour:
    if role is None:
        try:
            role = sagemaker.get_execution_role()
        except:
            logger.warning("User role not supplied and couldn't determine from environment")
    session = ProjectSession(project_id, role=role)
    if defaults.project_id and defaults.project_id != project_id and defaults.session:
        logger.info(f"Clearing previous default session for project '{defaults.project_id}'")
    defaults.project_id = project_id
    defaults.session = session
    logger.info(f"Working in project '{project_id}'")
    return session


def session_or_default(sess: Union[ProjectSession, None]=None, role: Union[str, None]=None):
    """Mostly-internal utility to return either the provided session or else a default"""
    if sess:
        return sess
    elif defaults.session:
        return defaults.session
    elif defaults.project_id:
        if role is None:
            try:
                role = sagemaker.get_execution_role()
            except:
                logger.warning("User role not supplied and couldn't determine from environment")
        defaults.session = ProjectSession(defaults.project_id, role=role)
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
