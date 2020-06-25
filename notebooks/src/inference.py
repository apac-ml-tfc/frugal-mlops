"""SageMaker inference wrapper for PyTorch TabNet"""

# Python Built-Ins:
import json
import logging
import os
import pickle

# External Dependencies:
import torch

logger = logging.getLogger()


def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    # TODO: Don't seem to be able to get these logs to show?
    print("Model loaded from pickle")
    logger.info("Model loaded from pickle")
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

    # If we expanded out a single-record request, contract the result back up to single-record:
    return result if is_batch_request else result[0]
