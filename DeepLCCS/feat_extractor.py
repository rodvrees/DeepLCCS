import os
import numpy as np
import pandas as pd
from collections import Counter
import random
import deeplcretrainer.cnn_functions
import sys
from sklearn.preprocessing import MinMaxScaler

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
    atom_radii[:, 0] = atom_counts[:, 0] * 1.70 #C
    atom_radii[:, 1] = atom_counts[:, 1] * 1.20 #H
    atom_radii[:, 2] = atom_counts[:, 2] * 1.55 #N
    atom_radii[:, 3] = atom_counts[:, 3] * 1.52 #O
    atom_radii[:, 4] = atom_counts[:, 4] * 1.80 #S
    atom_radii[:, 5] = atom_counts[:, 5] * 1.80 #P
    sum_radii = np.sum(atom_radii, axis=1)
    return sum_radii


def get_AA_vols(seq):
    length = len(seq)
    vol = 0
    for aa in seq:
        vol += aa_vol.loc[aa_vol["aa"] == aa, "vol"].values[0]
    vol_normalized = vol / (length * aa_vol.loc[aa_vol["aa"] == "G", "vol"].values[0])
    return length


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

#TODO: make general, use unimod API
def get_peptide_mass(peptide, modifications):
    mass = {
        "A": 71.03711,
        "C": 103.00919,
        "D": 115.02694,
        "E": 129.04259,
        "F": 147.06841,
        "G": 57.02146,
        "H": 137.05891,
        "I": 113.08406,
        "K": 128.09496,
        "L": 113.08406,
        "M": 131.04049,
        "N": 114.04293,
        "P": 97.05276,
        "Q": 128.05858,
        "R": 156.10111,
        "S": 87.03203,
        "T": 101.04768,
        "V": 99.06841,
        "W": 186.07931,
        "Y": 163.06333
    }
    mass = sum([mass[aa] for aa in peptide])
    if modifications != modifications:
        return mass
    else:
        modifications = modifications.split('|')[1::2]
        for mod in modifications:
            if mod == 'Oxidation':
                mass += 15.99491
            elif mod == 'Carbamidomethyl':
                mass += 57.02146
            elif mod == 'Acetyl':
                mass += 42.01056
            elif mod == 'Malonyl':
                mass += 86.000394
            elif mod == 'HexNAc':
                mass += 203.079373
            elif mod == 'Succinyl':
                mass += 100.016044
            elif mod == 'Glutarylation':
                mass += 114.031694
            elif mod == 'Formyl':
                mass += 27.994915
            elif mod == 'Propionyl':
                mass += 56.026215
            elif mod == 'Butyryl':
                mass += 70.041865
            elif mod == 'Crotonyl':
                mass += 68.026215
            elif mod == 'Methyl':
                mass += 14.01565
            elif mod == 'Dimethyl':
                mass += 28.0313
            elif mod == 'Trimethyl':
                mass += 42.04695
            elif mod == 'Biotin':
                mass += 226.077598
            elif mod == 'hydroxyisobutyryl':
                mass += 86.036779
            elif mod == 'GlyGly':
                mass += 114.042927
            elif mod == 'Phospho':
                mass += 79.966331
            elif mod == 'Citrullination':
                mass += 0.984016
            elif mod == 'Nitro':
                mass += 44.985078
            elif mod == 'Hydroxyproline':
                mass += 15.994915
        return mass


def get_global_feats(global_arr, df, mode="train", scaler=None):
    # Add charge to features
    global_feats = np.concatenate(
        (global_arr, df["charge"].values.reshape(-1, 1)), axis=1
    )
    # Add atom counts to features
    atom_counts = global_feats[:, 0:6]
    charge = global_feats[:, -1]
    # # Add sum of radii to features
    # sum_radii = get_atom_radii(atom_counts)
    # normalized_sum_radii = sum_radii / np.max(sum_radii)
    # Add volume to features
    df["lengths"] = df["seq"].apply(get_AA_vols)
    lengths = df["lengths"].values.reshape(-1, 1)
    # # Add AA ends composition to features
    # aa_ends = df["seq"].apply(get_atom_comp_ends)
    # aa_ends = np.stack(aa_ends.values)
    mass = df.apply(lambda x: get_peptide_mass(x['seq'], x['modifications']), axis=1)
    if mode == "train":
        scaler = MinMaxScaler()
        scaler.fit(mass.values.reshape(-1, 1))
    mass = scaler.transform(mass.values.reshape(-1, 1))
    H_rel_presence = df["seq"].apply(lambda x: x.count("H") / len(x))
    Bulky_rel_presence = df["seq"].apply(lambda x: (x.count("F") + x.count("W") + x.count("Y")) / len(x))
    Acid_rel_presence = df["seq"].apply(lambda x: (x.count("D") + x.count("E")) / len(x))
    KR_rel_presence = df["seq"].apply(lambda x: (x.count("K") + x.count("R")) / len(x))
    # Create global features
    global_feats = np.concatenate(
        (
            atom_counts,
            charge.reshape(-1, 1),
            # normalized_sum_radii.reshape(-1, 1),
            # #Add 0 for every data point
            # # np.zeros((len(normalized_sum_radii), 1)),
            lengths,
            # aa_ends,
            # np.zeros((len(aa_ends), 10)),
            mass,
            H_rel_presence.values.reshape(-1, 1),
            Bulky_rel_presence.values.reshape(-1, 1),
            Acid_rel_presence.values.reshape(-1, 1),
            KR_rel_presence.values.reshape(-1, 1),
            )
        ,
        axis=1,
    )
    if mode == "train":
        return global_feats, scaler
    else:
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
    ccs_df_train.to_csv("ccs_df_train.csv")
    ccs_df_test.to_csv("ccs_df_test.csv")
    return ccs_df_train, ccs_df_test

def get_features(ccs_df, dataset, architecture, num_lstm, info, log_level="info"):
    ccs_df_train, ccs_df_test = train_test_split(ccs_df)
    train_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_train, predict_ccs=True)
    test_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_test, predict_ccs=True)
    train_df["charge"] = ccs_df_train["charge"]
    test_df["charge"] = ccs_df_test["charge"]
    train_df["seq"] = ccs_df_train["seq"]
    test_df["seq"] = ccs_df_test["seq"]
    train_df["modifications"] = ccs_df_train["modifications"]
    test_df["modifications"] = ccs_df_test["modifications"]

    # train_df.to_csv(
    #     "./data/train_{}_{}_{}_{}.csv".format(
    #         dataset, architecture, num_lstm, info
    #     )
    # )

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

    global_feats_train, scaler = get_global_feats(X_train_global, train_df, mode="train")
    global_feats_test = get_global_feats(X_test_global, test_df, mode="test", scaler=scaler)

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

