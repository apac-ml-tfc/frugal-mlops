"""SageMaker inference functions for SKLearn random forest classifier"""

# Python Built-Ins:
import json
import os
import pickle
import random

# External Dependencies:
import joblib
import numpy as np


def model_fn(model_dir):
    """Generic model loader for model.joblib files"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def predict_fn(data, model):
    """Predict function override to return probabilities instead of labels"""
    return model.predict_proba(data)
