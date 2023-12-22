__author__ = ["Robbe Devreese, Robbin Bouwmeester"]
__credits__ = ["Robbe Devreese, Robbin Bouwmeester", "Prof. Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbe Devreese", "Robbin Bouwmeester"]
__email__ = ["robbe.devreese@ugent.be", "robbin.bouwmeester@ugent.be"]

# Import standard modules
import os
import argparse
import logging
import sys

# Import third party modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import local modules
from DeepLCCS.feat_extractor import get_features
from DeepLCCS.data_extractor import get_data
from DeepLCCS.model import compile_model, fit_model
from DeepLCCS.plot_results import plot_pred_results
from DeepLCCS.wandb_setup import start_wandb, stop_wandb
from DeepLCCS._argument_parser import parse_args
from DeepLCCS._exceptions import DeepLCCSError

logger = logging.getLogger(__name__)


def setup_logging(passed_level):
    log_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if passed_level.lower() not in log_mapping:
        raise ValueError(
            "Invalid log level. Should be one of the following: ",
            ", ".join(log_mapping.keys()),
        )

    logging.basisConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_mapping[passed_level.lower()],
    )


def main():
    "Main function for DeepLCCS"
    args = parse_args()

    setup_logging(args.log_level)

    try:
        run(**vars(args))
    except DeepLCCSError as e:
        logger.exception(e)
        sys.exit(1)


def run(
    dataset,
    epochs,
    batch_size,
    num_lstm,
    num_C_dense,
    num_concat_dense,
    v_split,
    optimizer,
    loss,
    metrics,
    activation,
    dropout_lstm,
    dropout_C_dense,
    dropout_concat_dense,
    architecture,
    kernel_size,
    strides,
    learning_rate,
    num_dense_layers,
    num_lstm_layers,
    info,
    log_level,
):
    """Run DeepLCCS training and valiation"""
    logger.info("Starting DeepLCCS")

    (
        ccs_df,
        X_train,
        global_feats_train,
        X_test,
        global_feats_test,
        ccs_df_train,
        ccs_df_test,
    ) = get_data(dataset, log_level, architecture, num_lstm, info)

#TODO: continue here, with wandb setup. Dont go conflating the wandb config, keep using the args