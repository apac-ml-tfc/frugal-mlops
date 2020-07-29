"""Lambda function to check if endpoint is updated after deployment/change

At the time of writing there's no createEndpoint.sync or updateEndpoint.sync actions in Step Functions
itself, so instead we can have Step Functions poll this Lambda, which checks whether the endpoint is
finished creating/updating/deleting/whatever yet and throws the specific EndpointUpdating error if not.
"""

# Python Built-Ins:
from datetime import date, datetime
import json

# External Dependencies:
import boto3


smclient = boto3.client("sagemaker")

DEFAULT_BUSY_STATES = set(["Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"])
DEFAULT_FAIL_STATES = set(["Failed"])

class EndpointUpdating(ValueError):
    """Error thrown if endpoint update still in progress (catch & retry this in your SFn)"""
    pass

class UpdateFailed(ValueError):
    """Error thrown if endpoint enters a FailState"""
    pass


def default_json_serializer(obj):
    """JSON serializer for objects not serializable by default json.dump"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError (f"Type {type(obj)} not serializable")


def handler(event, context):
    """Lambda handler to check if endpoint is updated after deployment/change

    Parameters
    ----------
    event.EndpointName : str
        Name of the endpoint to query
    event.BusyStates : List[str] (Optional)
        Optional override for set of recognised "busy" states raising EndpointUpdating
    event.TargetStates : List[str] (Optional)
        If supplied, entering any state other than BusyStates or TargetStates will trigger UpdateFailed. If
        not supplied, entering any state not listed in BusyStates or FailStates will be marked as completion.
    event.FailStates : List[str] (Optional)
        Optional override for set of specific fail states raising UpdateFailed
    """
    print(event)

    endpoint_name = event["EndpointName"]
    target_states = event.get("TargetStates")
    fail_states = event.get("FailStates", DEFAULT_FAIL_STATES)
    busy_states = event.get("BusyStates", DEFAULT_BUSY_STATES)

    endpoint_desc = smclient.describe_endpoint(EndpointName=endpoint_name)

    endpoint_status = endpoint_desc["EndpointStatus"]
    if endpoint_status in busy_states:
        raise EndpointUpdating(f"Endpoint {endpoint_name} is in status {endpoint_status}")
    elif target_states is not None and endpoint_status not in target_states:
        raise UpdateFailed(f"Endpoint {endpoint_name} is in status {endpoint_status}")
    elif endpoint_status in fail_states:
        raise UpdateFailed(f"Endpoint {endpoint_name} is in status {endpoint_status}")

    # Otherwise, success! The update is complete
    # We want to return a dict (so SFn treats it as an object rather than string), but one that's safe for
    # JSON serialization (datetime raises error by default):
    return json.loads(json.dumps(endpoint_desc, default=default_json_serializer))
