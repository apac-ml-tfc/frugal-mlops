"""SageMaker inference wrapper for PyTorch TabNet"""

# Python Built-Ins:
import json
import logging
import os
import pickle

# External Dependencies:
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import torch

logger = logging.getLogger()


def model_fn(model_dir):
    logger.info("Loading model metadata")
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        config = json.loads(f.read())

    model_path = os.path.join(model_dir, "tabnet.zip")
    logger.info(f"Loading model from {model_path}")
    model = TabNetClassifier() if config.get("modelType") == "classification" else TabNetRegressor()
    model.load_model(model_path)
    logger.info("Model loaded")

    return model


def predict_fn(input_data, model):
    # Note, the error when a user passes a CSV containing header strings is not super obvious (gets
    # deserialized to a string-like dtype instead of numeric), but trying to check with
    # np.issubdtype(..., np.number) just yields "TypeError: data type not understood" :-(
    # Watch out for:
    # TypeError: can't convert np.ndarray of type numpy.bytes_.

    is_batch_request = len(input_data.shape) >= 2
    if not is_batch_request:
        # PyTorch-TabNet complains about 1D input (i.e. single-record inference):
        input_data = input_data.unsqueeze(0)

    if callable(getattr(model, "predict_proba", None)):
        logger.info(
            "Predicting with probabilities on input_data of shape={}, dtype={}".format(
                input_data.shape,
                input_data.dtype,
            )
        )
        result = model.predict_proba(input_data)
    else:
        logger.info(
            f"Predicting scores only on input_data of shape={input_data.shape}, dtype={input_data.dtype}"
        )
        result = model.predict(input_data)

    # Normally if we wanted to offer a mixed single/multi-record request API, we'd probably check at this
    # point and return a single result rather than a nested array, if the request was single:
    # return result if is_batch_request else result[0]
    #
    # ...But this would be rendered as Score0\nScore1\nScore2 by the default CSV serializer, rather than
    # Score0,Score1,Score2 - and the default Model Monitor processor is fussy about CSV formatting:
    return result
