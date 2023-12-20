import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import deeplcretrainer
import deeplc
from collections import Counter
import os
from scipy.stats import pearsonr
import tensorflow.compat.v1 as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import argparse
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from DeepLCCS._argument_parser import parse_args
from models_bb.attention import EnhancedAttentionLayer
from DeepLCCS.data_extractor import get_data
from DeepLCCS.model_bb import (
    CNN_model_first_input,
    CNN_model_second_input,
    CNN_concatenate,
    LSTM_CNN_first_input,
    LSTM_CNN_second_input,
    LSTM_CNN_BiLSTM,
    LSTM_CNN_concatenate,
    LSTM_first_input,
    LSTM_second_input,
    LSTM_concatenate,
)

# import models_bb.APD_mimic as apd
import pickle


def main():
    dataset, args = parse_args()
    (
        X_train,
        global_feats_train,
        X_test,
        global_feats_test,
        ccs_df_train,
        ccs_df_test,
    ) = get_data(dataset, args=args)

    wandb.init(
        project="DeepLCCS",
        name="{}_{}_{}_{}".format(
            args.dataset, args.architecture, args.num_lstm, args.info
        ),
        save_code=True,
        config={
            "architecture": args.architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_lstm": args.num_lstm,
            "num_C_dense": args.num_C_dense,
            "num_concat_dense": args.num_concat_dense,
            "v_split": args.v_split,
            "optimizer": args.optimizer,
            "loss": args.loss,
            "metrics": args.metrics,
            "activation": args.activation,
            "dataset": args.dataset,
            "dropout_lstm": args.dropout_lstm,
            "dropout_C_dense": args.dropout_C_dense,
            "dropout_concat_dense": args.dropout_concat_dense,
            "info": args.info,
            "DEBUG": args.DEBUG,
            "kernel_size": args.kernel_size,
            "strides": args.strides,
            "learning_rate": args.learning_rate,
            "num_dense_layers": args.num_dense_layers,
            "num_lstm_layers": args.num_lstm_layers,
        },
    )

    config = wandb.config
    adam = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    if config.architecture == "CNN":
        # TODO This probably will not work, the output of each class should be the output, not a model object, look into this
        a = CNN_model_first_input(
            X_train, config.kernel_size, config.strides, padding="same"
        )
        b = CNN_model_second_input(config.num_C_dense, config.activation)
        model = CNN_concatenate(
            a.input,
            b.input,
            a.output,
            b.output,
            config.num_dense_layers,
            config.activation,
            config.num_concat_dense,
        )
        model.compile(optimizer=adam, loss=config.loss, metrics=config.metrics)

    if config.architecture == "CNN+LSTM":
        a = LSTM_CNN_first_input(
            X_train, config.kernel_size, config.strides, padding="same"
        )
        b = LSTM_CNN_second_input(config.num_C_dense, config.activation)
        c = LSTM_CNN_BiLSTM(X_train, config.num_lstm, config.dropout_lstm)
        model = LSTM_CNN_concatenate(
            a.input,
            b.input,
            c.input,
            a.output,
            b.output,
            c.output,
            config.num_dense_layers,
            config.activation,
            config.num_concat_dense,
        )
        model.compile(optimizer=adam, loss=config.loss, metrics=config.metrics)

    if config.architecture == "LSTM":
        a = LSTM_first_input(
            X_train, config.num_LSTM_layers, config.num_lstm, config.dropout_lstm
        )
        b = LSTM_second_input(config.num_C_dense, config.num_C_dense, config.activation)
        model = LSTM_concatenate(
            a.input,
            b.input,
            a.output,
            b.output,
            config.num_dense_layers,
            config.activation,
            config.num_concat_dense,
        )
        model.compile(optimizer=adam, loss=config.loss, metrics=config.metrics)

#TODO APD mimic

    if config.architecture == "CNN+LSTM":
        history = model.fit(
            (X_train, global_feats_train, X_train),
            ccs_df_train.loc[:, "tr"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.v_split,
            callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")],
        )
    else:
        # Fit the model on the training data
        history = model.fit(
            (X_train, global_feats_train),
            ccs_df_train.loc[:, "tr"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.v_split,
            callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")],
        )

    wandb.finish()

    # Predict CCS values test set
    ccs_df_test["LSTM_predictions"] = model.predict((X_test, global_feats_test))
    ccs_df_test.to_csv(
        "./preds/{}_{}_{}_{}.csv".format(
            args.dataset, args.architecture, args.num_lstm, args.info
        )
    )

    plot_results(ccs_df, ccs_df_test, ccs_df_train, args=args)


def plot_results(ccs_df, ccs_df_test, ccs_df_train, args={}):
    if len(ccs_df.index) < 1e4:
        set_alpha = 0.2
        set_size = 3
    else:
        set_alpha = 0.05
        set_size = 1

    # Scatter plot the observations on the test set against the predictions on the same set
    plt.scatter(
        ccs_df_test.loc[ccs_df_test["charge"] == 2, "tr"],
        ccs_df_test.loc[ccs_df_test["charge"] == 2, "LSTM_predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=2",
    )

    plt.scatter(
        ccs_df_test.loc[ccs_df_test["charge"] == 3, "tr"],
        ccs_df_test.loc[ccs_df_test["charge"] == 3, "LSTM_predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=3",
    )

    plt.scatter(
        ccs_df_test.loc[ccs_df_test["charge"] == 4, "tr"],
        ccs_df_test.loc[ccs_df_test["charge"] == 4, "LSTM_predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=4",
    )

    # Plot a diagonal the points should be one
    plt.plot([300, 1100], [300, 1100], c="grey")

    legend = plt.legend()

    for lh in legend.legendHandles:
        lh.set_sizes([25])
        lh.set_alpha(1)

    # Get the predictions and calculate performance metrics
    predictions = ccs_df_test["LSTM_predictions"]
    mare = round(
        sum(
            (abs(predictions - ccs_df_test.loc[:, "tr"]) / ccs_df_test.loc[:, "tr"])
            * 100
        )
        / len(predictions),
        3,
    )
    pcc = round(pearsonr(predictions, ccs_df_test.loc[:, "tr"])[0], 3)
    perc_95 = round(
        np.percentile(
            (abs(predictions - ccs_df_test.loc[:, "tr"]) / ccs_df_test.loc[:, "tr"])
            * 100,
            95,
        )
        * 2,
        2,
    )

    plt.title(f"LSTM - PCC: {pcc} - MARE: {mare}% - 95th percentile: {perc_95}%")

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.xlabel("Observed CCS (^2)")
    plt.ylabel("Predicted CCS (^2)")
    plt.savefig(
        "./figs/{}_{}_{}_{}.png".format(
            args.dataset, args.architecture, args.num_lstm, args.info
        ),
        dpi=300,
    )


if __name__ == "__main__":
    main()
