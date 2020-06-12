"""SageMaker inference wrapper for PyTorch TabNet"""

# Python Built-Ins:
import json
import logging
import os
import pickle


logger = logging.getLogger()


def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def predict_fn(input_data, model):
    return model.predict(input_data)
