"""Functions to enable and disable 'SageMaker Projects' for a user / execution role"""

# Python Built-Ins:
import logging

# External Dependencies:
import boto3

scclient = boto3.client("servicecatalog")
logger = logging.getLogger("smprojects")


def enable_sm_projects_for_role(studio_role_arn):
    """Enable SageMaker Projects for a SageMaker Execution Role

    This function assumes you've already run Boto SageMaker enable_sagemaker_servicecatalog_portfolio() for
    the account as a whole
    """
    portfolios_resp = scclient.list_accepted_portfolio_shares()

    portfolio_ids = set()
    for portfolio in portfolios_resp["PortfolioDetails"]:
        if portfolio["ProviderName"] == "Amazon SageMaker":
            portfolio_ids.add(portfolio["Id"])

    logger.info(f"Adding {len(portfolio_ids)} SageMaker SC portfolios to role {studio_role_arn}")
    for portfolio_id in portfolio_ids:
        scclient.associate_principal_with_portfolio(
            PortfolioId=portfolio_id,
            PrincipalARN=studio_role_arn,
            PrincipalType="IAM"
        )


def disable_sm_projects_for_role(studio_role_arn):
    """Enable SageMaker Projects for a SageMaker Execution Role

    This function assumes you've already run Boto SageMaker enable_sagemaker_servicecatalog_portfolio() for
    the account as a whole
    """
    portfolios_resp = scclient.list_accepted_portfolio_shares()

    portfolio_ids = set()
    for portfolio in portfolios_resp["PortfolioDetails"]:
        if portfolio["ProviderName"] == "Amazon SageMaker":
            portfolio_ids.add(portfolio["Id"])

    logger.info(f"Removing {len(portfolio_ids)} SageMaker SC portfolios from role {studio_role_arn}")
    for portfolio_id in portfolio_ids:
        response = scclient.disassociate_principal_from_portfolio(
            PortfolioId=portfolio_id,
            PrincipalARN=studio_role_arn,
        )
