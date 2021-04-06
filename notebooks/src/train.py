"""Train PyTorch TabNet"""

# Python Built-Ins:
import json
import logging
import os
import pickle
import random
import shutil
import tarfile

# External Dependencies:
import numpy as np
import torch
from torch.optim import lr_scheduler
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

# Local Dependencies:
import config
import data
from inference import *  # Needed for one-click deploy from Estimator


logger = logging.getLogger()


def set_seed(seed, use_gpus=True):
    """Seed all the random number generators we can think of for reproducibility"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_gpus:
            torch.cuda.manual_seed_all(seed)


def load_pretrained(pretrained_dir):
    """Load a pre-trained TabNet model from pretrained_dir (local folder)"""

    targz_files = list(filter(lambda f: f.lower().endswith(".tar.gz"), os.listdir(pretrained_dir)))
    if len(targz_files) > 1:
        logger.warning(f"Detected multiple .tar.gz files! Only the first will be extracted: {targz_files}")
    if len(targz_files) > 0:
        logger.info(f"Detected .tar.gz file... Extracting {targz_files[0]}")
        with tarfile.open(os.path.join(pretrained_dir, targz_files[0]), "r") as tar:
            tar.extractall(path=pretrained_dir)

    model_path = os.path.join(pretrained_dir, "tabnet.zip")
    if not os.path.isfile(model_path):
        logger.error(f"Expected PyTorch-TabNet model file {model_path} could not be found!")

    logger.info("Loading model...")
    model = TabNetPretrainer()
    model.load_model(model_path)
    logger.info("Loaded pre-trained model!")
    return model


def get_model(args):
    model_params = {
        "n_d": args.n_d,
        "n_a": args.n_a,
        "n_steps": args.n_steps,
        "gamma": args.gamma,
        "cat_idxs": args.cat_idxs,
        "cat_dims": args.cat_dims,
        "cat_emb_dim": args.cat_emb_dim,
        "n_independent": args.n_independent,
        "n_shared": args.n_shared,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "momentum": args.momentum,
        "clip_value": args.clip_value,
        "lambda_sparse": args.lambda_sparse,
        "mask_type": args.mask_type,
        # optimizer_fn unsupported
        "optimizer_params": dict(lr=args.lr),
        "scheduler_fn": (getattr(lr_scheduler, args.lr_scheduler) if args.lr_scheduler else None),
        "scheduler_params": args.lr_scheduler_params,
        # model_name see below
        #"saving_path": args.model_dir,
        # verbose unsupported
    }

    if args.model_type == "classification":
        ModelClass = TabNetClassifier
    elif args.model_type == "regression":
        ModelClass = TabNetRegressor
    elif args.model_type == "unsupervised":
        ModelClass = TabNetPretrainer
    else:
        raise ValueError(f"Unknown model_type {args.model_type} not in {config.MODEL_TYPES}")

    model_params = { k: v for k, v in model_params.items() if v is not None }
    return ModelClass(**model_params)


def copy_all_training_code_to_output(model_dir):
    """Copy training code to output, to enaable directly deploying via SageMaker SDK's Estimator.deploy()

    To enable directly deploying this model via SageMaker SDK's Estimator.deploy() (rather than needing to
    create a PyTorchModel with entry_point / source_dir args), we need to save any inference handler
    function code to model_dir/code. Here we compromise efficiency to the benefit of usage simplicity, by
    just copying the contents of this training code folder to the model/code folder for inference.
    """
    code_path = os.path.join(args.model_dir, "code")
    logger.info(f"Copying code to {code_path} for inference")
    for currpath, dirs, files in os.walk("."):
        for file in files:
            # Skip any filenames starting with dot:
            if file.startswith("."):
                continue
            filepath = os.path.join(currpath, file)
            # Skip any pycache or dot folders:
            if ((os.path.sep + ".") in filepath) or ("__pycache__" in filepath):
                continue
            relpath = filepath[len("."):]
            if relpath.startswith(os.path.sep):
                relpath = relpath[1:]
            outpath = os.path.join(code_path, relpath)
            logger.debug(f"Copying {filepath} to {outpath}")
            os.makedirs(outpath.rpartition(os.path.sep)[0], exist_ok=True)
            shutil.copy2(filepath, outpath)


def train(args):
    logger.info("Creating config and model")
    model = get_model(args)

    logger.info("Loading datasets")
    X_train, y_train = data.get_dataset(args.train, args)
    logger.info(f"X_train {X_train.shape}, y_train {y_train.shape}")
    if args.validation:
        X_val, y_val = data.get_dataset(args.validation, args)
        logger.info(f"X_val {X_val.shape}, y_val {y_val.shape}")
    else:
        X_val = None
        y_val = None

    logger.info("Collecting fit params")
    fit_params = {
        "X_train": X_train,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "virtual_batch_size": args.virtual_batch_size,
        "num_workers": args.num_workers,
        # weights unsupported
        #"weights": args.weights if args.model_type == "classification" else None,
        # loss_fn unsupported
        # drop_last unsupported
    }
    if args.pretrained:
        fit_params["from_unsupervised"] = load_pretrained(args.pretrained)
    if args.model_type != "unsupervised":
        fit_params["y_train"] = y_train
    else:
        fit_params["pretraining_ratio"] = args.pretraining_ratio
    if args.validation:
        fit_params["eval_name"] = ["validation"]
        fit_params["eval_set"] = [(X_val, y_val)] if args.model_type != "unsupervised" else [X_val]
        # (Could provide multiple sets, last one is used for early stopping)
        # See also eval_metrics=['?'], (Accuracy by default)

    fit_params = { k: v for k, v in fit_params.items() if v is not None }
    logger.info("Calling model.fit()...")
    model.fit(**fit_params)
    logger.info("model.fit() complete")

    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        f.write(json.dumps({
            "modelType": args.model_type,
        }))

    model.save_model(os.path.join(args.model_dir, "tabnet"))

    copy_all_training_code_to_output(args.model_dir)
    return model


if __name__ == "__main__":
    args = config.parse_args()

    for l in (logger, data.logger):
        config.configure_logger(l, args)

    logger.info("Loaded arguments: %s", args)
    logger.info("Starting!")
    set_seed(args.seed, use_gpus=args.num_gpus > 0)

    # Start training:
    train(args)
