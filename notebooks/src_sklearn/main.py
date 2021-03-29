"""SageMaker training/inference entry point for SKLearn random forest classifier"""

# Python Built-Ins:
import argparse
import json
import os
import random

# External Dependencies:
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Local Dependencies:
from inference import *  # Load in the inference functions for serving


def parse_args(cmd_args=None):
    """Parse config arguments from the command line, or cmd_args instead if supplied"""
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    parser = argparse.ArgumentParser(description="Train PyTorch-TabNet")

    ## Model parameters:
    parser.add_argument(
        "--n-estimators", type=int, default=hps.get("n-estimators", 100),
        help="The number of trees in the forest"
    )

    ## Data processing parameters:
    parser.add_argument(
        "--target", type=str, default=hps.get("target", "Target"),
        help="Name of the target column to predict."
    )

    ## Training process parameters:
    parser.add_argument("--seed", "--random-seed", type=int,
        default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)"
    )

    ## I/O Settings:
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--output-data-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args(args=cmd_args)

    # Accept numeric (index) target column specification:
    try:
        args.target = int(args.target)
    except ValueError:
        pass

    return args


def get_model(args):
    """Build the model architecture from the training job arguments"""
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        verbose=2,
    )
    return model


def get_dataset(channel, args):
    """Load a CSV dataset from file/folder `channel` to an X, y numpy pair"""
    if os.path.isdir(channel):
        contents = os.listdir(channel)
        if len(contents) == 1:
            data_path = os.path.join(channel, contents[0])
        else:
            csv_contents = list(filter(lambda s: s.endswith(".csv"), map(lambda s: s.lower(), contents)))
            if len(csv_contents) == 1:
                data_path = os.path.join(channel, csv_contents[0])
            else:
                raise ValueError(
                    "Channel folder {} must contain exactly one file or exactly one .csv. Got {}".format(
                        channel,
                        contents
                    )
                )
    elif os.path.isfile(channel):
        data_path = channel
    else:
        raise ValueError(f"Channel {channel} is neither file nor directory")

    print(f"Reading file {data_path}")
    df = pd.read_csv(data_path)
    print(f"Got shape {df.shape}")

    if isinstance(args.target, int):
        # args.target is a column index
        y = df.iloc[:, args.target]
        df.drop(df.columns[args.target], axis=1, inplace=True)
        return df.to_numpy(), y.to_numpy()
    elif isinstance(args.target, str):
        # args.target is a column name
        y = df[args.target]
        df.drop(args.target, axis=1, inplace=True)
        return df.to_numpy(), y.to_numpy()
    else:
        raise ValueError(
            f"args.target is neither str (column name) nor int (column index): Got {args.target}"
        )


def train(args):
    print("Creating config and model")
    model = get_model(args)

    print("Loading datasets")
    X_train, y_train = get_dataset(args.train, args)
    print(f"X_train {X_train.shape}, y_train {y_train.shape}")
    if args.validation:
        X_val, y_val = get_dataset(args.validation, args)
        print(f"X_val {X_val.shape}, y_val {y_val.shape}")
    else:
        X_val = None
        y_val = None

    print("Calling model.fit()...")
    model.fit(X_train, y_train)

    print("Saving model...")
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"model persisted at {model_path}")

    print("Calculating validation metrics...")
    validate(model, X_train, y_train, X_val, y_val)
    return model


def validate(model, X_train, y_train, X_val=None, y_val=None):
    """Calculate, log to console, and return, accuracy metrics for `model`"""

    metrics = { "train:accuracy": model.score(X_train, y_train) }
    if (X_val is not None) and (y_val is not None):
        metrics["validation:accuracy"] = model.score(X_val, y_val)

    print(" ".join(map(
        lambda name: f"{name}={metrics[name]};",
        metrics.keys(),
    )))
    return metrics


if __name__ == "__main__":
    args = parse_args()
    print(f"Loaded arguments: {args}")

    if args.seed:
        print(f"Seeding random number generators")
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("Starting!")
    train(args)
