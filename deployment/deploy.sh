#!/bin/bash

# Build the `project` and `sandbox` stacks and deploy the `demo` stack from which they can be provisioned
# NOTE: If `demo` is already provisioned, this will not update the Service Catalog products to new
# CloudFormation templates!

# Arguments:
STAGINGS3=$1
STACKNAME=$2
AWSPROFILE=$3

# Configuration:
DEMO_CF_TPLFILE=demo.yml
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

if [ -z "$STAGINGS3" ]
then
    echo -e "${RED}Argument Error:${NC} Deploy script usage"
    echo ""
    echo "./deploy-dev.sh {StagingS3} {StackName} [AWSProfile]"
    echo ""
    echo "StagingS3: Name of an existing S3 bucket for AWS SAM to build to and deploy from"
    echo "StackName: CloudFormation stack name to deploy for demo environment"
    echo "AWSProfile: (Optional) named AWS CLI login profile to use for requests"
    exit 1
fi

echo -e "Using '${CYAN}${AWSPROFILE}${NC}' as AWS profile"
echo -e "Using '${CYAN}${STAGINGS3}${NC}' as staging S3 bucket"
echo -e "Using '${CYAN}${STACKNAME}${NC}' as CloudFormation stack name"

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
echo "[Demo] Copying final CloudFormation template to S3..."
aws s3 cp $DEMO_CF_TPLFILE "s3://${STAGINGS3}/demo.yaml" --profile $AWSPROFILE

echo "Running SAM deploy..."
# Here we deliberately don't --no-fail-on-empty--changeset, because an empty changeset probably won't do what
# you're expecting (update the `project` or `sandbox` products in Service Catalog)
sam deploy \
    --template-file $DEMO_CF_TPLFILE \
    --stack-name $STACKNAME \
    --capabilities CAPABILITY_NAMED_IAM \
    --profile $AWSPROFILE \
    --parameter-overrides \
        DataSciProjectTemplateUrl=https://${STAGINGS3}.s3.amazonaws.com/project.yaml \
        DataSciSandboxTemplateUrl=https://${STAGINGS3}.s3.amazonaws.com/sandbox.yaml

echo -e "${CYAN}Full stack deployed!${NC}"
