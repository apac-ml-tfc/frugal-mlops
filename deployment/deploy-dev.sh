#!/bin/bash

# Build and deploy the `project` and `sandbox` stacks together against an existing SageMaker role.

# Arguments:
STAGINGS3=$1
PROJECTID=$2
ROLENAME=$3
EMAILADDRESS=$4
AWSPROFILE=$5

# Configuration:
# Project is an AWS SAM stack, others are currently plain CloudFormation:
PROJECT_SAM_TPLFILE=project.sam.yml
PROJECT_CF_TPLFILE=project.tmp.yml
SANDBOX_CF_TPLFILE=sandbox.yml

if [ -z "$AWSPROFILE" ]
then
    echo "AWSPROFILE not provided - using default"
    AWSPROFILE=default
fi

# Colorization (needs -e switch on echo, or to use printf):
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color (end)

if [ -z "$STAGINGS3" ] || [ -z "$PROJECTID" ] || [ -z "$ROLENAME" ] || [ -z "$EMAILADDRESS" ]
then
    echo -e "${RED}Argument Error:${NC} Deploy script usage"
    echo ""
    echo "./deploy-dev.sh {StagingS3} {ProjectId} {RoleName} {EmailAddress} [AWSProfile]"
    echo ""
    echo "StagingS3: Name of an existing S3 bucket for AWS SAM to build to and deploy from"
    echo "ProjectId: <=12 character lowercase ML project ID to deploy"
    echo "RoleName: Existing SageMaker execution role name to grant access to the project"
    echo "    and sandbox"
    echo "EmailAddress: Valid email address to send deployment pipeline approval requests"
    echo "AWSProfile: (Optional) named AWS CLI login profile to use for requests"
    exit 1
fi


echo -e "Using '${CYAN}${AWSPROFILE}${NC}' as AWS profile"
echo -e "Using '${CYAN}${STAGINGS3}${NC}' as staging S3 bucket"
echo -e "Using '${CYAN}${PROJECTID}${NC}' as project ID"
echo -e "Using '${CYAN}${ROLENAME}${NC}' as SageMaker role"
echo -e "Using '${CYAN}${EMAILADDRESS}${NC}' as manager email address"

# Exit if any build/deploy step fails:
set -e

echo "[Project] Running SAM build..."
sam build \
    --use-container \
    --template $PROJECT_SAM_TPLFILE \
    --profile $AWSPROFILE

echo "[Project] Running SAM package..."
sam package \
    --output-template-file $PROJECT_CF_TPLFILE \
    --s3-bucket $STAGINGS3 \
    --s3-prefix sam-project \
    --profile $AWSPROFILE

echo "[Project] Copying final CloudFormation template to S3..."
aws s3 cp $PROJECT_CF_TPLFILE "s3://${STAGINGS3}/project.yaml" --profile $AWSPROFILE
echo "[Sandbox] Copying final CloudFormation template to S3..."
aws s3 cp $SANDBOX_CF_TPLFILE "s3://${STAGINGS3}/sandbox.yaml" --profile $AWSPROFILE

echo "[Project] Running SAM deploy..."
sam deploy \
    --template-file $PROJECT_CF_TPLFILE \
    --stack-name ${PROJECTID}-project \
    --capabilities CAPABILITY_NAMED_IAM \
    --profile $AWSPROFILE \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
        ProjectId=$PROJECTID \
        ManagerEmail=$EMAILADDRESS \
        BaseSageMakerRoleName=$ROLENAME

# The 'aws cloudformation deploy' command has basically same interface - we'll use SAM for consistency:
echo "[Sandbox] Running SAM deploy..."
sam deploy \
    --template-file $SANDBOX_CF_TPLFILE \
    --stack-name ${PROJECTID}-sandbox \
    --capabilities CAPABILITY_NAMED_IAM \
    --profile $AWSPROFILE \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
        ProjectId=$PROJECTID \
        UserExecutionRole=$ROLENAME

echo -e "${CYAN}Full stack deployed!${NC}"
