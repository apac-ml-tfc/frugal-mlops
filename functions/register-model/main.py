"""Lambda function to register a model from sandbox to project"""

# Python Built-Ins:
from datetime import datetime
import io
import json
import os

# External Dependencies:
import boto3

# Local Dependencies:
import util

bucket_name = os.environ["PROJECT_BUCKET"]
bucket = boto3.resource("s3").Bucket(bucket_name)
smclient = boto3.client("sagemaker")


def bucket_and_key_from_s3_uri(s3uri):
    assert isinstance(s3uri, str) and s3uri.lower().startswith("s3://"), (
        f"s3uri must be a string beginning with 's3://': Got {s3uri}"
    )
    bucket, _, key = s3uri[len("s3://"):].partition("/")
    return bucket, key


def promote_model_container(container_def, new_data_uri, new_submit_uri=None):
    """Copy a container definition into a new environment"""
    result = json.loads(json.dumps(container_def))
    # TODO: Any changes to "Image"?
    result["ModelDataUrl"] = new_data_uri

    if "Environment" in result:
        env = result["Environment"]
        if new_submit_uri is not None:
            env["SAGEMAKER_SUBMIT_DIRECTORY"] = new_submit_uri
        elif env["SAGEMAKER_SUBMIT_DIRECTORY"]:
            raise ValueError("Model has a SAGEMAKER_SUBMIT_DIRECTORY but no replacement was provided")
    
        for k, v in env.items():
            if k != "SAGEMAKER_SUBMIT_DIRECTORY" and isinstance(v, str) and v.lower().startswith("s3://"):
                raise ValueError(
                    "Container definition contains non-ported S3 URI environment variable '{}'".format(
                        k
                    )
                )
    return result
    

def handler(event, context):
    print(event)

    training_job_name = event["TrainingJob"]["TrainingJobName"]
    trial_name = event["TrainingJob"]["ExperimentConfig"]["TrialName"]

    trial_desc = smclient.describe_trial(TrialName=trial_name)
    experiment_name = trial_desc["ExperimentName"]

    # TODO: Derive from something in input!
    target_model_name = util.append_timestamp("pipeline")

    folder = f"models/{experiment_name}/{trial_name}"

    bucket.upload_fileobj(io.BytesIO(json.dumps(event).encode("utf-8")), f"{folder}/request.json")

    # Copy the artifacts from sandbox to project
    # (Note training job output model.tar.gz might differ from registered model object model.tar.gz because
    # of additional code/ folder... We'll store both but use the latter for our model.)
    trainmodeltar_bucket_name, trainmodeltar_key = bucket_and_key_from_s3_uri(
        event["TrainingJob"]["ModelArtifacts"]["S3ModelArtifacts"]
    )
    bucket.copy(
        { "Bucket": trainmodeltar_bucket_name, "Key": trainmodeltar_key },
        f"{folder}/model-train.tar.gz"
    )
    modeltar_bucket_name, modeltar_key = bucket_and_key_from_s3_uri(
        event["Model"]["PrimaryContainer"]["ModelDataUrl"]
    )
    bucket.copy(
        { "Bucket": modeltar_bucket_name, "Key": modeltar_key },
        f"{folder}/model.tar.gz"
    )
    target_modeltar_uri = f"s3://{bucket_name}/{folder}/model.tar.gz"
    if "sagemaker_submit_directory" in event["TrainingJob"]["HyperParameters"]:
        traintar_bucket_name, traintar_key = bucket_and_key_from_s3_uri(
            # Hyperparams are JSON encoded to support non-string types (i.e. with "" wrapper)
            json.loads(event["TrainingJob"]["HyperParameters"]["sagemaker_submit_directory"])
        )
        bucket.copy(
            { "Bucket": traintar_bucket_name, "Key": traintar_key },
            f"{folder}/train-sourcedir.tar.gz"
        )
        target_traintar_uri = f"s3://{bucket_name}/{folder}/train-sourcedir.tar.gz"
    else:
        target_traintar_uri = None
    # TODO: Multi-container support
    if "SAGEMAKER_SUBMIT_DIRECTORY" in event["Model"]["PrimaryContainer"]["Environment"]:
        inftar_bucket_name, inftar_key = bucket_and_key_from_s3_uri(
            event["Model"]["PrimaryContainer"]["Environment"]["SAGEMAKER_SUBMIT_DIRECTORY"]
        )
        bucket.copy(
            { "Bucket": inftar_bucket_name, "Key": inftar_key },
            f"{folder}/inference.tar.gz"
        )
        target_inftar_uri = f"s3://{bucket_name}/{folder}/inference.tar.gz"
    else:
        target_inftar_uri = None

    # TODO: Check training job and model use same model.tar.gz, and maybe S3 modification datetime?

    create_model_response = smclient.create_model(
        ModelName=target_model_name,
        # Containers=[ 
        #     { 
        #         "ContainerHostname": "string",
        #         "Environment": { 
        #             "string" : "string" 
        #         },
        #         "Image": "string",
        #         "Mode": "string",
        #         "ModelDataUrl": "string",
        #         "ModelPackageName": "string"
        #     }
        # ],
        EnableNetworkIsolation=False,  # TODO
        ExecutionRoleArn=os.environ["PROJECT_MODEL_ROLE_ARN"],
        PrimaryContainer=promote_model_container(
            event["Model"]["PrimaryContainer"],
            target_modeltar_uri,
            target_inftar_uri,
        ),
        Tags=[
            { "Key": "Project", "Value": os.environ["PROJECT_ID"] },
            { "Key": "Pipeline-Status", "Value": "New" },
            { "Key": "ExperimentName", "Value": experiment_name },
            { "Key": "TrialName", "Value": trial_name },
            { "Key": "TrainingJobName", "Value": training_job_name },
        ],
        # TODO: VPC config
        #VpcConfig={ 
        #  "SecurityGroupIds": [ "string" ],
        #  "Subnets": [ "string" ]
        #}
    )
    return {
        "ModelArn": create_model_response["ModelArn"],
        "ModelName": target_model_name,
    }
