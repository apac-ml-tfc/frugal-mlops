"""SageMaker inference functions for SKLearn random forest classifier"""

# Python Built-Ins:
import argparse
import json
import os
import pickle
import random
import tarfile

# External Dependencies:
import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    """Generic model loader for model.joblib files"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def predict_fn(data, model):
    """Predict function override to return probabilities instead of labels"""
    return model.predict_proba(data)


####  OPTIONAL EXTENSION SECTION:
# The parts below are only needed for running the inference as a SageMaker Processing job instead of Batch
# Transform or Real-Time Inference. This is an edge case included for illustrative purposes, and can be
# ignored/deleted if you won't be using it!


def parse_inf_args(cmd_args=None):
    """Parse config arguments from the command line, or cmd_args instead if supplied"""
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    parser = argparse.ArgumentParser(description="Run inference as processing")

    ## Model parameters:
    parser.add_argument(
        "--n-estimators", type=int, default=hps.get("n-estimators", 100),
        help="The number of trees in the forest"
    )

    ## I/O Settings:
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/processing/model")
    )
    parser.add_argument("--input-data-dir", type=str,
        default=os.environ.get("SM_INPUT_DATA_DIR", "/opt/ml/processing/indata")
    )
    parser.add_argument("--result-dir", type=str,
        default=os.environ.get("SM_RESULT_DIR", "/opt/ml/processing/result")
    )

    args = parser.parse_args(args=cmd_args)
    return args


if __name__ == "__main__":
    args = parse_inf_args()
    print(f"Loaded arguments: {args}")

    # The model might be compressed (if training was in a SageMaker Training Job) or uncompressed
    # (if training was in a Processing job) - accommodate both:
    model_targz_path = os.path.join(args.model_dir, "model.tar.gz")
    if os.path.isfile(model_targz_path):
        print(f"Extracting model.tar.gz")
        with tarfile.open(model_targz_path, "r") as tar:
            tar.extractall(path=args.model_dir)

    print("Loading model")
    model = model_fn(args.model_dir)

    print("Transforming inputs")
    for currpath, dirs, files in os.walk(args.input_data_dir):
        for file in files:
            filepath = os.path.join(currpath, file)
            df = pd.read_csv(filepath)
            print(f"Got {len(df)} records from {filepath}")

            if len(df.columns) != model.n_features_:
                if len(df.columns) == model.n_features_ + 1:
                    print(f"Dropping final column to align data with model feature count")
                    df = df.iloc[:, :-1]
                else:
                    print(f"WARNING: Got {len(df.columns)} features in data vs {model.n_features_} in model")

            result = predict_fn(data=df, model=model)
            outpath = args.result_dir + filepath[len(args.input_data_dir):]
            print(f"Writing to {outpath}")
            np.savetxt(outpath, result, delimiter=",")

    print("Complete!")
