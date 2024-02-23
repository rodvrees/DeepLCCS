import pandas as pd
from collections import Counter
import random
import deeplcretrainer.cnn_functions
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

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

def get_features(ccs_df):
    ccs_df_train, ccs_df_test = train_test_split(ccs_df)
    train_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_train, predict_ccs=True)
    test_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_test, predict_ccs=True)
    train_df["charge"] = ccs_df_train["charge"]
    test_df["charge"] = ccs_df_test["charge"]
    train_df["seq"] = ccs_df_train["seq"]
    test_df["seq"] = ccs_df_test["seq"]
    train_df["modifications"] = ccs_df_train["modifications"]
    test_df["modifications"] = ccs_df_test["modifications"]

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

    train_data = {'X_train_AtomEnc' : X_train, 'X_train_DiAminoAtomEnc' : X_train_sum, 'X_train_GlobalFeatures' : X_train_global, 'X_train_OneHot' : X_train_hc, 'y_train' : y_train}
    test_data = {'X_test_AtomEnc' : X_test, 'X_test_DiAminoAtomEnc' : X_test_sum, 'X_test_GlobalFeatures' : X_test_global, 'X_test_OneHot' : X_test_hc, 'y_test' : y_test}

    return train_data, test_data

def main(info='DeepLC'):
    ccs_df = pd.read_csv("/home/robbe/DeepLCCS/data_clean/trainset.csv")
    train_data, test_data = get_features(ccs_df)

    for key in train_data:
        pickle.dump(train_data[key], open(f"../data_clean/{key}-{info}.pickle", "wb"))
    for key in test_data:
        pickle.dump(test_data[key], open(f"../data_clean/{key}-{info}.pickle", "wb"))

if __name__ == "__main__":
    main()
