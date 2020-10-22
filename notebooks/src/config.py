"""SageMaker configuration parsing for PyTorch-TabNet

For the most part pretty faithful to documented pytorch-tabnet parameter interface
"""

# Python Built-Ins:
import argparse
import json
import logging
import os
import sys

# External Dependencies:
import torch

MODEL_TYPES=("classification", "regression")

def configure_logger(logger, args):
    """Configure a logger's level and handler (since base container already configures top level logging)"""
    consolehandler = logging.StreamHandler(sys.stdout)
    consolehandler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(consolehandler)
    logger.setLevel(args.log_level)


def boolean_hyperparam(raw):
    """Boolean argparse type for convenience in SageMaker
    SageMaker HPO supports categorical variables, but doesn't have a specific type for booleans -
    so passing `command --flag` to our container is tricky but `command --arg true` is easy.
    Using argparse with the built-in `type=bool`, the only way to set false would be to pass an
    explicit empty string like: `command --arg ""`... which looks super weird and isn't intuitive.
    Using argparse with `type=boolean_hyperparam` instead, the CLI will support all the various
    ways to indicate 'yes' and 'no' that you might expect: e.g. `command --arg false`.
    """
    valid_false = ("0", "false", "n", "no", "")
    valid_true = ("1", "true", "y", "yes")
    raw_lower = raw.lower()
    if raw_lower in valid_false:
        return False
    elif raw_lower in valid_true:
        return True
    else:
        raise argparse.ArgumentTypeError(
        f"'{raw}' value for case-insensitive boolean hyperparam is not in valid falsy "
        f"{valid_false} or truthy {valid_true} value list"
    )

def list_hyperparam(raw):
    """Basic comma-separated list argparse type for convenience in SageMaker
    No escaping of commas supported, no conversion from raw string type (see list_hyperparam_withparser).
    """
    return [] if raw is None else raw.split(",")

def list_hyperparam_withparser(parser, default=None, ignore_error=False):
    # Define separate functions, rather than using logic, as it's easy & Python try/except can be expensive
    def unsafe_mapper(val):
        result = parser(val)
        return default if result is None else result

    def safe_mapper(val):
        try:
            result = parser(val)
            return default if result is None else result
        except Exception:
            return default

    return lambda raw: list(map(safe_mapper if ignore_error else unsafe_mapper, list_hyperparam(raw)))


def parse_args(cmd_args=None):
    """Parse config arguments from the command line, or cmd_args instead if supplied"""
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    parser = argparse.ArgumentParser(description="Train PyTorch-TabNet")

    ## Network parameters:
    parser.add_argument(
        "--model-type", type=str, default=hps.get("model-type", "classification"),
        help=f"Model type selected in the list: {', '.join(MODEL_TYPES)}"
    )
    parser.add_argument(
        "--n-d", type=int, default=hps.get("n-d", 8),
        help="Width of the decision prediction layer. Bigger values give more capacity to the model with "
        "the risk of overfitting. Values typically range from 8-64."
    )
    parser.add_argument(
        "--n-a", type=int, default=hps.get("n-a", 8),
        help="Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a "
        "good choice."
    )
    parser.add_argument(
        "--n-steps", type=int, default=hps.get("n-steps", 3),
        help="Number of steps in the architecture - usually 3-10."
    )
    parser.add_argument(
        "--gamma", type=float, default=hps.get("gamma", 1.3),
        help="Coefficient for feature reusage in the masks. A value close to 1 will make mask selection "
        "least correlated between layers. Values range from 1.0-2.0."
    )
    parser.add_argument(
        "--n-independent", type=int, default=hps.get("n-independent", 2),
        help="Number of independent Gated Linear Units layers at each step. Usual range 1-5."
    )
    parser.add_argument(
        "--n-shared", type=int, default=hps.get("n-shared", 2),
        help="Number of shared Gated Linear Units at each step. Usual range 1-5."
    )
    parser.add_argument(
        "--epsilon", type=float, default=hps.get("epsilon", 1e-15),
        help="If you need help, you don't need to change me."
    )
    parser.add_argument(
        "--lambda-sparse", type=float, default=hps.get("lambda-sparse", 1e-3),
        help="The extra sparsity loss coefficient proposed in the original paper. The bigger the "
        "coefficient is, the sparser the model will be in terms of feature selection. Depending on the "
        "difficulty of the problem, reducing this value could help."
    )

    ## Data processing parameters:
    parser.add_argument(
        "--target", type=str, default=hps.get("target", "Target"),
        help="Name of the target column to predict."
    )
    parser.add_argument(
        "--cat-idxs", type=list_hyperparam_withparser(int),
        default=hps.get("cat-idxs", []),
        help="List of categorical feature indexes."
    )
    parser.add_argument(
        "--cat-dims", type=list_hyperparam_withparser(int),
        default=hps.get("cat-dims", []),
        help="List of dimensions (number of unique values) of each categorical feature."
    )
    parser.add_argument(
        "--cat-emb-dim", type=list_hyperparam_withparser(int, default=1),
        default=hps.get("cat-emb-dim", [1]),
        help="List of categorical embedding sizes for each categorical feature (Or a single number to share)"
    )

    ## Training process parameters:
    parser.add_argument("--seed", "--random-seed", type=int,
        default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)"
    )
    parser.add_argument(
        "--momentum", type=float, default=hps.get("momentum", 0.02),
        help="Momentum for batch normalization - typical range 0.01-0.4."
    )
    parser.add_argument("--lr", "--learning-rate", type=float,
        default=hps.get("lr", hps.get("learning-rate", 5e-5)),
        help="Learning rate (main training cycle)"
    )
    parser.add_argument(
        "--clip-value", type=float, default=hps.get("clip-value"),
        help="Optional gradient value clipping"
    )
    # optimizer_fn param not supported
    # scheduler_fn param not supported
    # scheduler_params param not supported
    # model_name param not necessary
    # saving_path param not necessary
    # verbose param see log-level
    # device_name param not necessary
    parser.add_argument("--max-epochs", type=int, default=hps.get("max-epochs", 200),
        help="Maximum number of epochs for training"
    )
    parser.add_argument("--patience", type=int, default=hps.get("patience", 15),
        help="Number of consecutive epochs without improvement before early stopping"
    )
    # weights param not supported
#     parser.add_argument("--weights", type=???intordict?, default=hps.get("weights", 0),
#         help="Only applicable for classification problems: sampling parameter 0=no sampling, "
#         "1=automated sampling with inverse class occurrences, dict class->weights for manual."
#     )
    # loss_fn param not supported
    parser.add_argument("--batch-size", type=int, default=hps.get("batch-size", 1024),
        help="Number of examples per batch. Large batch sizes are recommended."
    )
    parser.add_argument("--virtual-batch-size", type=int, default=hps.get("virtual-batch-size", 128),
        help="Size of mini-batches for 'Ghost Batch Normalization'"
    )
    # drop_last param not supported

    # Resource Management:
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", 0),
        help="Number of GPUs to use in training."
    )
    parser.add_argument("--num-workers", "-j", type=int,
        default=hps.get("num-workers", max(0, int(os.environ.get("SM_NUM_CPUS", 0)) - 2)),
        help="Number of data workers: set higher to accelerate data loading, if CPU and GPUs are powerful"
    )

    # I/O Settings:
    parser.add_argument("--log-level", default=hps.get("log-level", logging.INFO),
        help="Log level (per Python specs, string or int)."
    )
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--output-data-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args(args=cmd_args)

    ## Post-argparse validations & transformations:

    # Set up log level: (Convert e.g. "20" to 20 but leave "DEBUG" alone)
    try:
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    # Note basicConfig has already been called by our parent container, so calling it won't do anything.
    logger = logging.getLogger("config")
    configure_logger(logger, args)

    # Check model-type recognised:
    if args.model_type not in MODEL_TYPES:
        parser.error(f"--model-type must be one of {MODEL_TYPES}")

    # Accept numeric (index) target column specification:
    try:
        args.target = int(args.target)
    except ValueError:
        pass

    # Categorical feature indexes/etc consistency:
    n_cat_idxs = len(args.cat_idxs)
    n_cat_emb_dims = len(args.cat_emb_dim)
    if n_cat_idxs == 0:
        args.cat_idxs = None
        args.cat_emb_dim = None
    elif n_cat_idxs != n_cat_emb_dims:
        if n_cat_emb_dims == 1:
            # Apply fixed embedding dimension to all categorical features:
            args.cat_emb_dim = [args.cat_emb_dim[0] for _ in range(n_cat_idxs)]
        else:
            parser.error(
                f"Mismatch: Got {n_cat_idxs} --cat-idxs but {n_cat_emb_dims} --cat-emb-dims"
            )

    if args.num_gpus and not torch.cuda.is_available():
        parser.error(
            f"Got --num-gpus {args.num_gpus} but torch says cuda is not available: Cannot use GPUs"
        )

    return args
