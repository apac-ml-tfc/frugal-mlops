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

## Getting Started

This repository only templatizes **part** of the required setup: the **ML project stack**.

First you'll need to:

- [Set up SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) and make a note of your user's *Execution Role*. Note that if you're following this lab along in **AWS Event Engine**, you'll need to onboard with IAM - not SSO.
- (Unless you'd like to [use the default bucket](https://sagemaker.readthedocs.io/en/stable/api/utility/session.html#sagemaker.session.Session.default_bucket)), create an S3 bucket that your *SageMaker Execution Role* has full access to: This will be your sandbox area.
- Open SageMaker Studio, start a "System Terminal" from the Launcher page, and `git clone https://github.com/apac-ml-tfc/frugal-mlops`

Click the button below to launch the **ML project stack** in **us-east-1** (N. Virginia):

[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://us-east-1.console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=frugalmlops&templateURL=https://public-frugal-mlops-us-east-1.s3.amazonaws.com/package.yaml)

Once your CloudFormation stack and SageMaker Studio environment are set up, you're ready to follow through the notebooks in the [/notebooks](notebooks) folder.
