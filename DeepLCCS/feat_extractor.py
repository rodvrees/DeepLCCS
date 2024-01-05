import os
import numpy as np
import pandas as pd
from collections import Counter
import random
import deeplcretrainer.cnn_functions
import sys

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

filepath = os.path.dirname(os.path.realpath(__file__))
aa_comp = pd.read_csv(filepath + "/aa_comp.csv")
aa_vol = pd.read_csv(filepath + "/aa_vol.csv")

def get_atom_radii(atom_counts):
    atom_radii = np.zeros((atom_counts.shape[0], 6))
    atom_radii[:, 0] = atom_counts[:, 0] * 170 #C
    atom_radii[:, 1] = atom_counts[:, 1] * 120 #H
    atom_radii[:, 2] = atom_counts[:, 2] * 155 #N
    atom_radii[:, 3] = atom_counts[:, 3] * 152 #O
    atom_radii[:, 4] = atom_counts[:, 4] * 180 #S
    atom_radii[:, 5] = atom_counts[:, 5] * 180 #P
    sum_radii = np.sum(atom_radii, axis=1)
    return sum_radii


def get_AA_vols(seq):
    length = len(seq)
    vol = 0
    for aa in seq:
        vol += aa_vol.loc[aa_vol["aa"] == aa, "vol"].values[0]
    vol_normalized = vol / (length * aa_vol.loc[aa_vol["aa"] == "G", "vol"].values[0])
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
        (
            atom_counts,
            charge.reshape(-1, 1),
            normalized_sum_radii.reshape(-1, 1),
            vols,
            aa_ends,
        ),
        axis=1,
    )
    return global_feats

#TODO make test_split a parameter
def train_test_split(ccs_df):
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
    return ccs_df_train, ccs_df_test

def get_features(ccs_df, dataset, architecture, num_lstm, info, log_level="info"):
    ccs_df_train, ccs_df_test = train_test_split(ccs_df)
    train_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_train, predict_ccs=True)
    test_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_test, predict_ccs=True)
    train_df["charge"] = ccs_df_train["charge"]
    test_df["charge"] = ccs_df_test["charge"]
    train_df["seq"] = ccs_df_train["seq"]
    test_df["seq"] = ccs_df_test["seq"]

    train_df.to_csv(
        "./data/train_{}_{}_{}_{}.csv".format(
            dataset, architecture, num_lstm, info
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

    if log_level == "debug":
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

