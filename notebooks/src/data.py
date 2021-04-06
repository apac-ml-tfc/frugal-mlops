"""SageMaker data loading utilities for PyTorch TabNet"""

# Python Built-Ins:
import logging
import os

# External Dependencies:
import pandas as pd

logger = logging.getLogger("data")

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

    logger.info(f"Reading file {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Got shape {df.shape}")

    if isinstance(args.target, int):
        # args.target is a column index
        y = df.iloc[:, args.target]
        df.drop(df.columns[args.target], axis=1, inplace=True)
        return df.to_numpy(), y.to_numpy()
    elif isinstance(args.target, str):
        # args.target is a column name
        if args.target in df:
            y = df[args.target]
            df.drop(args.target, axis=1, inplace=True)
            return df.to_numpy(), y.to_numpy()
        elif args.model_type == "unsupervised":
            return df.to_numpy(), None
        else:
            raise ValueError(
                f"Target column name '{args.target}' not in {data_path}, and model_type not 'unsupervised'"
            )
    else:
        raise ValueError(
            f"args.target is neither str (column name) nor int (column index): Got {args.target}"
        )
