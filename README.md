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

Click the button below to launch the solution stack in **us-east-1** (N. Virginia):

[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://us-east-1.console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=AllStoreDemo&templateURL=https://public-lunar-lander-apac-us-east-1.s3.amazonaws.com/package.yaml)
