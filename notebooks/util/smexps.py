"""Convenience functions for SageMaker Experiments"""

# Python Built-Ins:
from typing import Dict, List, Union
import warnings

# External Dependencies:
from botocore import exceptions as botoexceptions
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker


def create_or_load_experiment(*args, **kwargs):
    """Create a SageMaker Experiemnt, or load existing if one of the same name already exists
    
    Params / return values are as per smexperiments Experiment.create() / Experiment.load()
    """
    try:
        return Experiment.create(*args, **kwargs)
    except botoexceptions.ClientError as err:
        errcontent = err.response["Error"]
        if errcontent["Code"] == "ValidationException" and "must be unique" in errcontent["Message"]:
            # Experiment already exists:
            # Need to extract a subset of arguments to send to the load() call.
            experiment_name = kwargs.get("experiment_name")
            if experiment_name is None:
                if len(args):
                    experiment_name = args[0]
                else:
                    raise ValueError("Couldn't determine experiment_name to load existing Experiment")
            warnings.warn(f"Using existing Experiment '{experiment_name}'")
            return Experiment.load(
                experiment_name,
                sagemaker_boto_client=kwargs.get("sagemaker_boto_client")
            )
        else:
            # Some other problem:
            raise err

def trial_component_has_parents(trial_component_name) -> Union[bool, List[Dict[str, str]]]:
    """Check if a trial component has any parents (Trials, Experiments)

    Returns
    -------
    parents :
        False or a list of 1 or more {"TrialName", "ExperimentName"} pairs
    """
    # Only way to access "Parents" seems to be via the search API, so that's what we'll do:
    components = list(TrialComponent.search(
        experiment_search.SearchExpression(
            filters=[
                experiment_search.Filter(
                    name="TrialComponentName",
                    operator=experiment_search.Operator.EQUALS,
                    value="TrialComponent-2020-06-12-151324-fbrs"
                )
            ]
        ),
        max_results=2
    ))
    n_components = len(components)
    if n_components == 0:
        raise ValueError(f"TrialComponentName {trial-component_name} not found")
    elif n_components > 1:
        # This should not happen as the API should enforce uniqueness:
        raise ValueError(f"Found multiple components matching TrialComponentName={trial-component_name}")

    return components[0].parents if len(components[0].parents) else False


def delete_trial_and_components(trial):
    """Delete a Trial and any associated TrialComponents that don't also belong to other trials."""
    trial = Trial.load(trial) if isinstance(trial, str) else trial
    for component in trial.list_trial_components():
        # TODO: Verify performantness (does remove_trial_component call TrialComponent.load anyway?)
        trial.remove_trial_component(component.trial_component_name)
        if not trial_component_has_parents(component.trial_component_name):
            # TODO: Swallow error if the component can't be deleted after being detached?
            TrialComponent.load(component.trial_component_name).delete()
    trial.delete()


def delete_experiment_and_trials(experiment):
    """Delete an Experiment and any associated Trials and TrialComponents that aren't used elsewhere.

    Note that while TrialComponents can belong to multiple Trials; a Trial has exactly one parent experiment
    so we don't see the same orphan checking logic.
    """
    experiment = Experiment.load(experiment) if isinstance(experiment, str) else experiment
    for trial in experiment.list_trials():
        delete_trial_and_components(trial.trial_name)

    experiment.delete()
