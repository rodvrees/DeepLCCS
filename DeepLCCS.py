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

from models_bb.attention import EnhancedAttentionLayer 

# import models_bb.APD_mimic as apd
import pickle

vol_dict = {
    "A": 88.6,
    "B": 0.0,
    "O": 0.0,
    "X": 0.0,
    "J": 0.0,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140,
}


aa_comp = pd.read_csv("./aa_comp.csv")

def get_atom_radii(atom_counts):
    atom_radii = np.zeros((atom_counts.shape[0], 6))
    atom_radii[:, 0] = atom_counts[:, 0] * 170
    atom_radii[:, 1] = atom_counts[:, 1] * 120
    atom_radii[:, 2] = atom_counts[:, 2] * 155
    atom_radii[:, 3] = atom_counts[:, 3] * 152
    atom_radii[:, 4] = atom_counts[:, 4] * 180
    atom_radii[:, 5] = atom_counts[:, 5] * 180
    sum_radii = np.sum(atom_radii, axis=1)
    return sum_radii


def get_AA_vols(seq):
    length = len(seq)
    vol = 0
    for aa in seq:
        vol += vol_dict[aa]
    vol_normalized = vol / (length * vol_dict["G"])
    return vol_normalized


def get_atom_comp_ends(seq):
    """
    Get the atom composition of the first and last two amino acids in a seq and put it in a vector.

    Parameters:
    - seq (str): A string representing the amino acid seq.

    Example:
    get_atom_comp_ends("PGPVLVDLPK")
    returns
    array([7, 14, 2, 2, 0, 11, 23, 3, 2, 0])
    """
    # Get the amino acid composition for the first and last two amino acids
    first_aa = aa_comp.loc[aa_comp["aa"] == seq[0]].iloc[0, 1:].values
    second_aa = aa_comp.loc[aa_comp["aa"] == seq[1]].iloc[0, 1:].values
    second_to_last_aa = aa_comp.loc[aa_comp["aa"] == seq[-2]].iloc[0, 1:].values
    last_aa = aa_comp.loc[aa_comp["aa"] == seq[-1]].iloc[0, 1:].values

    # Sum the compositions of the first two and last two amino acids
    sum_first_two = np.sum([first_aa, second_aa], axis=0)
    sum_last_two = np.sum([second_to_last_aa, last_aa], axis=0)

    # Concatenate the results to get the final vector
    result = np.concatenate([sum_first_two, sum_last_two])
    return result.astype(int)


def get_global_feats(global_arr, df):
    # Add charge to features
    global_feats = np.concatenate(
        (global_arr, df["charge"].values.reshape(-1, 1)), axis=1
    )
    # Add atom counts to features
    atom_counts = global_feats[:, 0:6]
    charge = global_feats[:, -1]
    # Add sum of radii to features
    sum_radii = get_atom_radii(atom_counts)
    normalized_sum_radii = sum_radii / np.max(sum_radii)
    # Add volume to features
    df["vol"] = df["seq"].apply(get_AA_vols)
    vols = df["vol"].values.reshape(-1, 1)
    # Add AA ends composition to features
    aa_ends = df["seq"].apply(get_atom_comp_ends)
    aa_ends = np.stack(aa_ends.values)
    # Create global features
    global_feats = np.concatenate(
        (atom_counts, charge.reshape(-1, 1), normalized_sum_radii.reshape(-1, 1), vols, aa_ends),
        axis=1,
    )
    return global_feats


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DeepLCCS model.")
    parser.add_argument(
        "--dataset", type=str, default="sample", help="full, sample or path to csv file"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to train the model"
    )
    parser.add_argument(
        "--num_lstm", type=int, default=24, help="Number of LSTM units")
    parser.add_argument(
        "--num_C_dense", type=int, default=5, help="Number of dense units for charge"
    )
    parser.add_argument(
        "--num_concat_dense",
        type=list,
        default=[64, 32],
        help="Number of dense units after concatenation",
    )
    parser.add_argument("--v_split", type=float, default=0.1, help="Validation split")
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use"
    )
    parser.add_argument(
        "--loss", type=str, default="mean_squared_error", help="Loss function to use"
    )
    parser.add_argument(
        "--metrics", type=list, default=["mean_absolute_error"], help="Metrics to use"
    )
    parser.add_argument(
        "--activation", type=str, default="relu", help="Activation function to use"
    )
    parser.add_argument(
        "--dropout_lstm", type=float, default=0.0, help="Dropout for LSTM"
    )
    parser.add_argument(
        "--dropout_C_dense", type=float, default=0.0, help="Dropout for dense layers"
    )
    parser.add_argument(
        "--dropout_concat_dense",
        type=list,
        default=[0.0, 0.0],
        help="Dropout for dense layers after concatenation",
    )
    parser.add_argument(
        "--architecture", type=str, default="LSTM", help="Architecture to use"
    )
    parser.add_argument(
        "--info", type=str, default="", help="Extra info to add to the run name"
    )
    parser.add_argument(
        "--DEBUG", type=bool, default=False, help="Debug mode"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=10, help="Kernel size for CNN"
    )
    parser.add_argument(
        "--strides", type=int, default=1, help="Strides for CNN"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    args = parser.parse_args()

    dataset = args.dataset

    return dataset, args


def get_features(ccs_df, args={}):
    X_matrix_count = pd.DataFrame(ccs_df["seq"].apply(Counter).to_dict()).fillna(0.0).T
    # Get all the index identifiers
    all_idx = list(X_matrix_count.index)
    random.seed(42)

    # Shuffle the index identifiers so we can randomly split them in a testing and training set
    random.shuffle(all_idx)

    # Select 90 % for training and the remaining 10 % for testing
    train_idx = all_idx[0 : int(len(all_idx) * 0.9)]
    test_idx = all_idx[int(len(all_idx) * 0.9) :]

    # Get the train and test indices and point to new variables
    ccs_df_train = ccs_df.loc[train_idx, :]
    ccs_df_test = ccs_df.loc[test_idx, :]

    train_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_train, predict_ccs=True)
    test_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_test, predict_ccs=True)
    train_df["charge"] = ccs_df_train["charge"]
    test_df["charge"] = ccs_df_test["charge"]
    train_df["seq"] = ccs_df_train["seq"]
    test_df["seq"] = ccs_df_test["seq"]

    train_df.to_csv(
        "./data/train_{}_{}_{}_{}.csv".format(
            args.dataset, args.architecture, args.num_lstm, args.info
        )
    )

    (
        X_train,
        X_train_sum,
        X_train_global,
        X_train_hc,
        y_train,
    ) = deeplcretrainer.cnn_functions.get_feat_matrix(train_df)
    (
        X_test,
        X_test_sum,
        X_test_global,
        X_test_hc,
        y_test,
    ) = deeplcretrainer.cnn_functions.get_feat_matrix(test_df)

    global_feats_train = get_global_feats(X_train_global, train_df)
    global_feats_test = get_global_feats(X_test_global, test_df)

    if args.DEBUG:
        ccs_df.to_csv("debug.csv")
        global_feats_train.tofile("global_feats_train.csv", sep=",")
        

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return (
        X_train,
        global_feats_train,
        X_test,
        global_feats_test,
        ccs_df_train,
        ccs_df_test,
    )


def main():
    dataset, args = parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if dataset == "full":
        ccs_df = pd.read_csv("./data/peprec_CCS.csv")
    elif dataset == "sample":
        ccs_df = pd.read_csv("./data/ccs_sample.csv")
    else:
        if os.path.isfile(dataset):
            ccs_df = pd.read_csv(dataset)
        else:
            FileNotFoundError(f"File {dataset} not found.")
    if args.DEBUG:
        ccs_df = ccs_df.sample(100, random_state=42)

    try:
        if dataset == "full":
        
            X_train = pickle.load(open("X_train_full.pickle", "rb"))
            global_feats_train = pickle.load(open("global_feats_train_full.pickle", "rb"))
            X_test = pickle.load(open("X_test_full.pickle", "rb"))
            global_feats_test = pickle.load(open("global_feats_test_full.pickle", "rb"))
            ccs_df_train = pickle.load(open("ccs_df_train_full.pickle", "rb"))
            ccs_df_test = pickle.load(open("ccs_df_test_full.pickle", "rb"))
        
        elif dataset == "sample":
            X_train = pickle.load(open("X_train_sample.pickle", "rb"))
            global_feats_train = pickle.load(open("global_feats_train_sample.pickle", "rb"))
            X_test = pickle.load(open("X_test_sample.pickle", "rb"))
            global_feats_test = pickle.load(open("global_feats_test_sample.pickle", "rb"))
            ccs_df_train = pickle.load(open("ccs_df_train_sample.pickle", "rb"))
            ccs_df_test = pickle.load(open("ccs_df_test_sample.pickle", "rb"))

    except IOError:
        (
            X_train,
            global_feats_train,
            X_test,
            global_feats_test,
            ccs_df_train,
            ccs_df_test,
        ) = get_features(ccs_df, args=args)

        if args.dataset == "full":
            pickle.dump(X_train, open("X_train_full.pickle", "wb"))
            pickle.dump(global_feats_train, open("global_feats_train_full.pickle", "wb"))
            pickle.dump(X_test, open("X_test_full.pickle", "wb"))
            pickle.dump(global_feats_test, open("global_feats_test_full.pickle", "wb"))
            pickle.dump(ccs_df_train, open("ccs_df_train_full.pickle", "wb"))
            pickle.dump(ccs_df_test, open("ccs_df_test_full.pickle", "wb"))
        
        elif args.dataset == "sample":
            pickle.dump(X_train, open("X_train_sample.pickle", "wb"))
            pickle.dump(global_feats_train, open("global_feats_train_sample.pickle", "wb"))
            pickle.dump(X_test, open("X_test_sample.pickle", "wb"))
            pickle.dump(global_feats_test, open("global_feats_test_sample.pickle", "wb"))
            pickle.dump(ccs_df_train, open("ccs_df_train_sample.pickle", "wb"))
            pickle.dump(ccs_df_test, open("ccs_df_test_sample.pickle", "wb"))

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
        },
    )

    config = wandb.config

    if config.architecture == "CNN":
        adam = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        input_a = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
        a = Conv1D(
            filters=1,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding="same",
        )(input_a)
        # a = Conv1D(
        #    filters=128,
        #    kernel_size=5,
        #    strides=1,
        #    padding="same",
        # )(a)

        a = Flatten()(a)
        a = tf.keras.Model(inputs=input_a, outputs=a)

        input_b = tf.keras.Input(shape=(19,))
        b = tf.keras.layers.Dense(config.num_C_dense, activation=config.activation)(
            input_b
        )
        b = tf.keras.Model(inputs=input_b, outputs=b)

        c = tf.keras.layers.concatenate([a.output, b.output], axis=-1)
        c = tf.keras.layers.Dense(64, activation=config.activation)(c)
        c = tf.keras.layers.Dense(64, activation=config.activation)(c)
        c = tf.keras.layers.Dense(64, activation=config.activation)(c)
        c = tf.keras.layers.Dense(64, activation=config.activation)(c)
        c = tf.keras.layers.Dense(64, activation=config.activation)(c)
        c = tf.keras.layers.Dense(1, activation=config.activation)(c)
        # Create the final model
        model = tf.keras.Model(inputs=[a.input, b.input], outputs=c)
        model.compile(
            optimizer=adam, loss=config.loss, metrics=config.metrics
        )

    if config.architecture == "CNN+LSTM":
        input_a = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
        # Bidirectional LSTM
        a = Conv1D(
            filters=128,
            kernel_size=4,
            strides=4,
            padding="same",
        )(input_a)
        a = Conv1D(
            filters=128,
            kernel_size=4,
            strides=1,
            padding="same",
        )(a)
        MaxPooling1D(pool_size=2)(a)
        a = Conv1D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding="same",
        )(a)
        a = Conv1D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding="same",
        )(a)
        MaxPooling1D(pool_size=2)(a)

        a = Flatten()(a)
        a = tf.keras.Model(inputs=input_a, outputs=a)

        input_b = tf.keras.Input(shape=(19,))
        b = tf.keras.layers.Dense(config.num_C_dense, activation=config.activation)(
            input_b
        )
        b = tf.keras.Model(inputs=input_b, outputs=b)

        input_c = tf.keras.Input(shape=(None, X_train.shape[2]))
        # Bidirectional LSTM
        c = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                config.num_lstm, return_sequences=False, dropout=config.dropout_lstm
            )
        )(input_c)
        # a = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.num_lstm, return_sequences=False))(a)
        # a = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.num_lstm, dropout=config.dropout_lstm))(a)
        c = tf.keras.Model(inputs=input_c, outputs=c)

        d = tf.keras.layers.concatenate([a.output, b.output, c.output], axis=-1)
        d = tf.keras.layers.Dense(32, activation=config.activation)(d)
        d = tf.keras.layers.Dense(32, activation=config.activation)(d)
        d = tf.keras.layers.Dense(1, activation=config.activation)(d)
        # Create the final model
        model = tf.keras.Model(inputs=[a.input, b.input, c.input], outputs=d)
        model.compile(
            optimizer=config.optimizer, loss=config.loss, metrics=config.metrics
        )

    if config.architecture == "LSTM":
        adam = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        input_a = tf.keras.Input(shape=(None, X_train.shape[2]))
        # Bidirectional LSTM
        a = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                config.num_lstm, return_sequences=False, dropout=config.dropout_lstm
            )
        )(input_a)

        a = tf.keras.Model(inputs=input_a, outputs=a)

        # Input for global features
        input_b = tf.keras.Input(shape=(19,))
        # Dense layers for global features
        b = tf.keras.layers.Dense(config.num_C_dense, activation=config.activation)(
            input_b
        )
        b = tf.keras.Model(inputs=input_b, outputs=b)

        # Concatenate the two layers
        c = tf.keras.layers.concatenate([a.output, b.output], axis=-1)

        # Dense layers after concatenation
        c = tf.keras.layers.Dense(
            config.num_concat_dense[0], activation=config.activation
        )(c)
        c = tf.keras.layers.Dense(
            config.num_concat_dense[1], activation=config.activation
        )(c)
        c = tf.keras.layers.Dense(1, activation=config.activation)(c)
        # Create the final model
        model = tf.keras.Model(inputs=[a.input, b.input], outputs=c)
        model.compile(
            optimizer=adam, loss=config.loss, metrics=config.metrics
        )

    if config.architecture == "embedding":
        model = apd.embedding_model()
        model.compile(
            optimizer=config.optimizer, loss=config.loss, metrics=config.metrics
        )

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