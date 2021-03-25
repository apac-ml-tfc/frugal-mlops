# Frugal MLOps with Amazon SageMaker

Good, automated ML model management and governance is a business problem crossing several different domains: The solution lies in effective **business processes, not just tools** - and will likely span a range of technologies in search of the right tool for each job.

To help navigate the crowd of possibilities, this repo presents an **opinionated example** of how some MLOps goals can be met using Amazon SageMaker and other services on the AWS cloud. Even if the described workflow doesn't work for you, it should introduce some useful tools and control points that could be tailored for your team.


## Guiding Principles

We generally observe the following **significant differences** between MLOps and traditional DevOps:

- Because **data is a dependency**, we'd like tools to track, test, and version data; maintaining a controlled and traceable mapping from data to solution, just like we normally would from dependency library versions to solution.
- Because **data science is experimental**, we'll do what we can to streamline the interactive exploration and experimentation process for data scientists. We'll use the features of SageMaker to deliver traceability even in interactive workflows.
- Because **training models can be resource-intensive**, we'll focus on enforcing control points around existing artifacts - rather than building pipelines that rebuild artifacts on each promotion.

These differences and an aim towards [frugality](https://en.wiktionary.org/wiki/frugality) are reflected in the **high-level decisions** for this repository's architecture:

- **Serverless first**: Prefer serverless orchestration (e.g. [AWS Lambda](https://aws.amazon.com/lambda/) and [Step Functions](https://aws.amazon.com/step-functions/)), to build pipelines where we pay for execution on demand - rather than having to manage cluster capacity as we might with [Kubeflow](https://www.kubeflow.org/) or similar tools.
- **Artifacts over commits**: Git is great for code, and VCS-backed pipelines are great for code-based workflows... But interactive, data-driven experimentation doesn't sit so well producing tidy code repositories with useful DAGs. We'll combine VCS-based flows with artifact-oriented flows to boost productivity and remove friction.

## Solution Architecture

This repository sets out 4 deployable concepts as shown below, each backed by a CloudFormation stack:

- An **environment**: Including a provisioned SageMaker Studio domain, CloudFormation helper apparatus for the remaining stacks, and the other three stacks packaged as deployable **products** in [AWS Service Catalog](https://aws.amazon.com/servicecatalog/)
- A **project**: The shared infrastructure of a data science project including artifact stores and automation pipelines
- A **data science user**: The deployable infrastructure for onboarding an individual data scientist
- A **sandbox**: Construct *Linking* a data scientist to a project with access and personal sandbox stores - and access granted to the project's protected resources

**TODO: DIAGRAM TO BE UPDATED!**

![Environment, Project and Sandbox Overview](img/architecture-overview.png)

A SageMaker Execution Role is assumed to be unique to a data scientist, because this is the main level at which notebook permissions can be controlled. A data scientist is assumed to access multiple projects through one role, rather than for example forcing them to assume separate role sessions inside the SageMaker environment.

In principle, the scientist reads data from the project and performs experiments in their sandbox environment (SageMaker Studio or notebook instance, and sandbox bucket(s)). Successful models and artifacts are then published back up to the project via helper utilities and deployment pipeline(s).


## Getting Started

### Pre-requisites

- An AWS Account, targeting a region which has not yet been onboarded to SageMaker Studio (no domain present)
- A local environment with:
  - Administrative credentials for the account (e.g. set up via `aws configure`)
  - The [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html) and [Docker](https://www.docker.com/products/docker-desktop) installed
  - The `make` and `envsubst` command line utilities installed

### Build and Deploy

First, build the SAM stacks and deploy the base environment:

```sh
# Navigate to the deployment directory
cd deployment

# Build all the stacks and create the ML Environment stack
make all
```

Once this ML Environment stack finishes deploying successfully, you can either deploy the other components manually as a typical use case would:

- Deploy a 'User' from the [AWS Service Catalog console](https://console.aws.amazon.com/servicecatalog/home?#products)
- Deploy a 'Project' either from Service Catalog or the 'Projects > Organizational Templates' section in SageMaker Studio
- Deploy a 'Sandbox' from Service Catalog, linking your username to your deployed project

...Or continue setup through the CLI utilities:

```
make deploy.project STACK_NAME=frugal-project
make deploy.user STACK_NAME=frugal-user
make deploy.sandbox STACK_NAME=frugal-sandbox
```

Refer to the `Makefile` and `make` help message for more details on supported options in the CLI!
