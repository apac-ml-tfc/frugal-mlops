"""Extra Boto3 utilities/simplifications for SageMaker"""

# Python Built-Ins:
from datetime import datetime
from dateutil.tz import tzlocal

# External Dependencies:
import boto3
import botocore


def assumed_role_session(role_arn: str, base_session: botocore.session.Session = None):
    """Create a boto3 Session with an assumed role (by ARN) different from the default sagemaker role

    From https://stackoverflow.com/a/45834847
    """
    base_session = base_session or boto3.session.Session()._session
    fetcher = botocore.credentials.AssumeRoleCredentialFetcher(
        client_creator=base_session.create_client,
        source_credentials=base_session.get_credentials(),
        role_arn=role_arn,
        extra_args={
            # "RoleSessionName": "atempsession" # set this if you want something non-default
        }
    )
    creds = botocore.credentials.DeferredRefreshableCredentials(
        method="assume-role",
        refresh_using=fetcher.fetch_credentials,
        time_fetcher=lambda: datetime.now(tzlocal()),
    )
    botocore_session = botocore.session.Session()
    botocore_session._credentials = creds
    return boto3.Session(botocore_session = botocore_session)

def s3uri_to_bucket_and_key(s3uri: str):
    if not s3uri.lower().startswith("s3://"):
        return ValueError(f"s3uri must start with s3:// Got: {s3uri}")
    bucket, _, key = s3uri[len("s3://"):].partition("/")
    return bucket, key
