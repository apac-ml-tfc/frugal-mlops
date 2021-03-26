#### MLOps Demo

STACK_NAME?="frugal-mlops"
#DEPLOYMENT_BUCKET_NAME?="UNDEFINED"
DEPLOYMENT_BUCKET_PREFIX?=""
SAM_BUILD_EXTRA_ARGS?=""

MANAGER_EMAIL?="donotreply@example.com"
ML_PROJECT_ID?="frugalmlops"
ML_USER_ID?="frugaluser"
SC_PRINCIPAL?="role/Admin"
USE_EXISTING_SC_ROLES?="yes"
USER_DEFAULT_CODE_REPO?="https://github.com/apac-ml-tfc/frugal-mlops"

target: ##=> Without a target, show the help message
	$(info ${HELP_MESSAGE})
	@exit 0

init: ##=> Install OS deps and dev tools for buildserver environments
	$(info [*] Bootstrapping CI system...)
	@$(MAKE) _install_os_packages

build.project: ##=> Build and package the ML Project stack only
	$(info [*] Building ML Project SAM stack...)
	sam build \
			--use-container \
			--template project.sam.yaml \
			$(SAM_BUILD_EXTRA_ARGS) && \
		sam package \
			--s3-bucket $${DEPLOYMENT_BUCKET_NAME} \
			--s3-prefix $${DEPLOYMENT_BUCKET_PREFIX}sam \
			--output-template-file project.tmp.yaml && \
		aws s3 cp project.tmp.yaml \
			s3://$${DEPLOYMENT_BUCKET_NAME}/$${DEPLOYMENT_BUCKET_PREFIX}project.cf.yaml

build.user: ##=> Build and package the Data Scientist stack only
	$(info [*] Building Data Science User SAM stack...)
	sam build \
			--use-container \
			--template datascientist.sam.yaml \
			$(SAM_BUILD_EXTRA_ARGS) && \
		sam package \
			--s3-bucket $${DEPLOYMENT_BUCKET_NAME} \
			--s3-prefix $${DEPLOYMENT_BUCKET_PREFIX}sam \
			--output-template-file datascientist.tmp.yaml && \
		aws s3 cp datascientist.tmp.yaml \
			s3://$${DEPLOYMENT_BUCKET_NAME}/$${DEPLOYMENT_BUCKET_PREFIX}datascientist.cf.yaml

build.sandbox: ##=> Build and package the ML Sandbox stack only
	$(info [*] Building ML Sandbox SAM stack...)
	sam build \
			--use-container \
			--template sandbox.yaml \
			$(SAM_BUILD_EXTRA_ARGS) && \
		sam package \
			--s3-bucket $${DEPLOYMENT_BUCKET_NAME} \
			--s3-prefix $${DEPLOYMENT_BUCKET_PREFIX}sam \
			--output-template-file sandbox.tmp.yaml && \
		aws s3 cp sandbox.tmp.yaml \
			s3://$${DEPLOYMENT_BUCKET_NAME}/$${DEPLOYMENT_BUCKET_PREFIX}sandbox.cf.yaml

build.environment: ##=> Build and package the ML Environment stack only
	$(info [*] Building ML Environment SAM stack...)
	sam build \
			--use-container \
			--template environment.sam.yaml \
			$(SAM_BUILD_EXTRA_ARGS) && \
		sam package \
			--s3-bucket $${DEPLOYMENT_BUCKET_NAME} \
			--s3-prefix $${DEPLOYMENT_BUCKET_PREFIX}sam \
			--output-template-file environment-presub.tmp.yaml && \
		export MLProjectCFURL=https://$${DEPLOYMENT_BUCKET_NAME}.s3.`aws configure get region`.amazonaws.com/$${DEPLOYMENT_BUCKET_PREFIX}project.cf.yaml && \
		export DataScienceUserCFURL=https://$${DEPLOYMENT_BUCKET_NAME}.s3.`aws configure get region`.amazonaws.com/$${DEPLOYMENT_BUCKET_PREFIX}datascientist.cf.yaml && \
		export MLSandboxCFURL=https://$${DEPLOYMENT_BUCKET_NAME}.s3.`aws configure get region`.amazonaws.com/$${DEPLOYMENT_BUCKET_PREFIX}sandbox.cf.yaml && \
		envsubst '$$MLProjectCFURL,$$DataScienceUserCFURL,$$MLSandboxCFURL' <environment-presub.tmp.yaml >environment.tmp.yaml && \
		aws s3 cp environment.tmp.yaml \
			s3://$${DEPLOYMENT_BUCKET_NAME}/$${DEPLOYMENT_BUCKET_PREFIX}environment.cf.yaml
# (We use envsubst to replace the Project CF template URL in the environment template with the built &
# staged version)

build: ##=> Build SAM template & assets to CloudFormation on S3
	$(info [*] Building all AWS SAM stacks...)
	@$(MAKE) build.user
	@$(MAKE) build.project
	@$(MAKE) build.sandbox
	@$(MAKE) build.environment

create: ##=> Create environment stack (only)
	$(info [*] Deploying ML Environment...)
	# This option disables rollback for debugging, versus SAM `deploy`
	aws cloudformation create-stack \
			--template-body file://environment.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
			--disable-rollback \
			--parameters \
			  ParameterKey=UseExistingSageMakerServiceCatalogRoles,ParameterValue=$(USE_EXISTING_SC_ROLES) \
				ParameterKey=SCPrincipal,ParameterValue=$(SC_PRINCIPAL)
			# (disable rollback for debugging)
			# TODO: Support --parameters through MAKE?

create.project: ##=> Create project stack (only), assuming ML environment ready
	$(info [*] Deploying ML Project...)
	# This option disables rollback for debugging, versus SAM `deploy`
	aws cloudformation create-stack \
			--template-body file://project.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
			--disable-rollback \
			--parameters ParameterKey=ManagerEmail,ParameterValue=$(MANAGER_EMAIL) \
				ParameterKey=SageMakerProjectName,ParameterValue=$(ML_PROJECT_ID) \
				ParameterKey=ProjectId,ParameterValue=$(ML_PROJECT_ID)
			# TODO: Support --parameters through MAKE?

create.user: ##=> Create data science user stack (only), assuming ML environment ready
	$(info [*] Deploying Data Science User...)
	# This option disables rollback for debugging, versus SAM `deploy`
	aws cloudformation create-stack \
			--template-body file://datascientist.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
			--disable-rollback \
			--parameters ParameterKey=StudioUserName,ParameterValue=$(ML_USER_ID) \
				ParameterKey=DefaultCodeRepo,ParameterValue=$(USER_DEFAULT_CODE_REPO)
			# TODO: Support --parameters through MAKE?

create.sandbox: ##=> Create ML sandbox stack (only)
	$(info [*] Deploying ML Sandbox...)
	# This option disables rollback for debugging, versus SAM `deploy`
	aws cloudformation create-stack \
			--template-body file://sandbox.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
			--disable-rollback \
			--parameters ParameterKey=ProjectId,ParameterValue=$(ML_PROJECT_ID) \
				ParameterKey=StudioUserName,ParameterValue=$(ML_USER_ID)
			# TODO: Support --parameters through MAKE?

deploy: ##=> Deploy (create or update) ML Environment
	$(info [*] Deploying ML Environment...)
	sam deploy \
			--template-file environment.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
			--no-fail-on-empty-changeset \
			--parameter-overrides SCPrincipal=$(SC_PRINCIPAL) \
			  UseExistingSageMakerServiceCatalogRoles=$(USE_EXISTING_SC_ROLES)

deploy.project: ##=> Deploy (create or update) ML Project, assuming ML Environment ready
	$(info [*] Deploying ML Project...)
	sam deploy \
			--template-file project.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
			--no-fail-on-empty-changeset \
			--parameter-overrides ManagerEmail=$(MANAGER_EMAIL) \
				SageMakerProjectName=$(ML_PROJECT_ID) \
				ProjectId=$(ML_PROJECT_ID)

deploy.user: ##=> Deploy (create or update) ML User, assuming ML Environment ready
	$(info [*] Deploying Data Science User...)
	sam deploy \
			--template-file datascientist.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
			--no-fail-on-empty-changeset \
			--parameter-overrides StudioUserName=$(ML_USER_ID) \
				DefaultCodeRepo=$(USER_DEFAULT_CODE_REPO)

deploy.sandbox: ##=> Deploy (create or update) ML Sandbox stack, assuming Project & User
	$(info [*] Deploying ML Sandbox...)
	sam deploy \
			--template-file sandbox.tmp.yaml \
			--stack-name $(STACK_NAME) \
			--capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
			--no-fail-on-empty-changeset \
			--parameter-overrides ProjectId=$(ML_PROJECT_ID) \
				StudioUserName=$(ML_USER_ID)


all: ##=> Build all stacks and create the ML Environment stack
	@$(MAKE) build
	@$(MAKE) create

all.complete: ##=> Build all stacks and (try to) deploy the full stack set
	@$(MAKE) build
	@$(MAKE) deploy
	@$(MAKE) deploy.user STACK_NAME=frugal-user
	@$(MAKE) deploy.project STACK_NAME=frugal-project
	@$(MAKE) deploy.sandbox STACK_NAME=frugal-sandbox

delete: ##=> Delete services
	$(info [*] Deleting stack...)
	aws cloudformation delete-stack --stack-name $(STACK_NAME)


## HELPERS:

_install_os_packages:
	$(info [*] Installing jq...)
	yum install jq -y
	$(info [*] Checking currently installed Python version...)
	python3 --version
	$(info [*] Upgrading Python SAM CLI and CloudFormation linter to the latest version...)
	python3 -m pip install --upgrade --user cfn-lint aws-sam-cli
	npm -g install aws-cdk

define HELP_MESSAGE

	Environment variables:

	STACK_NAME: "frugal-mlops"
		Description: CloudFormation stack name to deploy/redeploy to. Remember to reset this parameter
		  if making multiple successive calls e.g. create.project, create.user!
	DEPLOYMENT_BUCKET_NAME: "UNDEFINED"
		Description: Amazon S3 bucket for staging built SAM Lambda bundles and assets
	DEPLOYMENT_BUCKET_PREFIX: ""
		Description: For publishing to a prefix in your deployment bucket, instead of root. Should end
		  in a slash if set.
	SAM_BUILD_EXTRA_ARGS: ""
		Description: Extra arguments to pass to AWS SAM build, if necessary

	MANAGER_EMAIL: "donotreply@example.com"
		Description: (For Project stack) Email address to receive deployment approval requests from
		  the pipeline.
	ML_PROJECT_ID: "frugalmlops"
		Description: (For Project & Sandbox stacks) ML Project ID (used as a prefix for resources)
	ML_USER_ID: "frugaluser"
		Description: (For User & Sandbox stacks) SageMaker Studio Username
	SC_PRINCIPAL: "role/Admin"
		Description: (For Environment stack) A valid IAM principal in your account to grant access to
		  the created service catalog portfolio (e.g. the identity you log in to the AWS Console with)
	USE_EXISTING_SC_ROLES: "yes"
		Description: (For Environment stack) Set 'no' to create SageMaker Service Catalog service
		roles in your account, or 'yes' if they exist already.
	USER_DEFAULT_CODE_REPO: "https://github.com/apac-ml-tfc/frugal-mlops"
		Description: (For User stack) The (optional) git repository to clone in to the created
		  SageMaker Studio user's home folder.


	Common usage:


	...::: Build all stacks and create the ML Environment stack (incl. service catalog) :::...
	$ make all

	...::: Build and package all SAM based stacks :::...
	$ make build

	...::: Build a particular SAM stack (environment, user, project, sandbox) :::...
	$ make build.environment

	...::: Deploy or re-deploy the ML Environment stack, waiting for completion :::...
	$ make deploy

	...::: Deploy or re-deploy a particular SAM stack :::...
	$ make deploy.project STACK_NAME=frugal-project

	...::: Create the ML Environment stack with rollback disabled (for debugging) :::...
	$ make create

	...::: Create a specific stack with rollback disabled (for debugging) :::...
	$ make create.user STACK_NAME=frugal-user

	...::: Deploy or re-deploy the ML Environment stack (incl. service catalog) :::...
	$ make all

	...::: (Experimental) Deploy all stacks with some default stack names :::...
	$ make all.complete

	...::: Bootstrap environment with necessary tools like SAM CLI, cfn-lint, etc. :::...
	$ make init

	...::: Delete a particular stack :::...
	$ make delete STACK_NAME=frugal-sandbox
endef
