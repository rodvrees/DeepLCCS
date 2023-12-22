import pandas as pd
import os
import pickle
from DeepLCCS.feat_extractor import get_features


def get_data(dataset, architecture, num_lstm, info, log_level="info"):
    if dataset == "full":
        ccs_df = pd.read_csv("./data/peprec_CCS.csv")
    elif dataset == "sample":
        ccs_df = pd.read_csv("./data/ccs_sample.csv")
    else:
        if os.path.isfile(dataset):
            ccs_df = pd.read_csv(dataset)
        else:
            FileNotFoundError(f"File {dataset} not found.")
    if log_level == "debug":
        ccs_df = ccs_df.sample(100, random_state=42)

    try:
        if dataset == "full":
            X_train = pickle.load(open("X_train_full.pickle", "rb"))
            global_feats_train = pickle.load(
                open("global_feats_train_full.pickle", "rb")
            )
            X_test = pickle.load(open("X_test_full.pickle", "rb"))
            global_feats_test = pickle.load(open("global_feats_test_full.pickle", "rb"))
            ccs_df_train = pickle.load(open("ccs_df_train_full.pickle", "rb"))
            ccs_df_test = pickle.load(open("ccs_df_test_full.pickle", "rb"))

        elif dataset == "sample":
            X_train = pickle.load(open("X_train_sample.pickle", "rb"))
            global_feats_train = pickle.load(
                open("global_feats_train_sample.pickle", "rb")
            )
            X_test = pickle.load(open("X_test_sample.pickle", "rb"))
            global_feats_test = pickle.load(
                open("global_feats_test_sample.pickle", "rb")
            )
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
        ) = get_features(ccs_df, dataset, architecture, num_lstm, info)

        if dataset == "full":
            pickle.dump(X_train, open("X_train_full.pickle", "wb"))
            pickle.dump(
                global_feats_train, open("global_feats_train_full.pickle", "wb")
            )
            pickle.dump(X_test, open("X_test_full.pickle", "wb"))
            pickle.dump(global_feats_test, open("global_feats_test_full.pickle", "wb"))
            pickle.dump(ccs_df_train, open("ccs_df_train_full.pickle", "wb"))
            pickle.dump(ccs_df_test, open("ccs_df_test_full.pickle", "wb"))

        elif dataset == "sample":
            pickle.dump(X_train, open("X_train_sample.pickle", "wb"))
            pickle.dump(
                global_feats_train, open("global_feats_train_sample.pickle", "wb")
            )
            pickle.dump(X_test, open("X_test_sample.pickle", "wb"))
            pickle.dump(
                global_feats_test, open("global_feats_test_sample.pickle", "wb")
            )
            pickle.dump(ccs_df_train, open("ccs_df_train_sample.pickle", "wb"))
            pickle.dump(ccs_df_test, open("ccs_df_test_sample.pickle", "wb"))

    return (
        ccs_df,
        X_train,
        global_feats_train,
        X_test,
        global_feats_test,
        ccs_df_train,
        ccs_df_test,
    )
