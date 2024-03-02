import pandas as pd
import os
import pickle
from DeepLCCS.feat_extractor import get_features
import logging

logger = logging.getLogger(__name__)


def get_data(dataset, architecture, num_lstm, info, log_level="info"):
    if dataset == "full":
        logger.info("Loading full dataset")
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
            X_train = pickle.load(open("/home/robbe/DeepLCCS/data/X_train_full-trainsetwithmassesscaled.pickle", "rb"))
            global_feats_train = pickle.load(
                open("/home/robbe/DeepLCCS/data/global_feats_train_full-trainsetwithmassesscaled.pickle", "rb")
            )
            X_test = pickle.load(open("/home/robbe/DeepLCCS/data/X_test_full-trainsetwithmassesscaled.pickle", "rb"))
            global_feats_test = pickle.load(open("/home/robbe/DeepLCCS/data/global_feats_test_full-trainsetwithmassesscaled.pickle", "rb"))
            ccs_df_train = pickle.load(open("/home/robbe/DeepLCCS/data/ccs_df_train_full-trainsetwithmassesscaled.pickle", "rb"))
            ccs_df_test = pickle.load(open("/home/robbe/DeepLCCS/data/ccs_df_test_full-trainsetwithmassesscaled.pickle", "rb"))

        elif dataset == "sample":
            X_train = pickle.load(open("/home/robbe/DeepLCCS/data/X_train_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "rb"))
            global_feats_train = pickle.load(
                open("/home/robbe/DeepLCCS/data/global_feats_train_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "rb")
            )
            X_test = pickle.load(open("/home/robbe/DeepLCCS/data/X_test_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "rb"))
            global_feats_test = pickle.load(
                open("/home/robbe/DeepLCCS/data/global_feats_test_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "rb")
            )
            ccs_df_train = pickle.load(open("/home/robbe/DeepLCCS/data/ccs_df_train_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "rb"))
            ccs_df_test = pickle.load(open("/home/robbe/DeepLCCS/data/ccs_df_test_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "rb"))
            logger.info("Training and test split already exist, loading them...")

        else:
            X_train = pickle.load(open("/home/robbe/DeepLCCS/data/X_train_trainset_all_features_onehot.pickle", "rb"))
            global_feats_train = pickle.load(open("/home/robbe/DeepLCCS/data/global_feats_train_trainset_all_features_onehot.pickle", "rb"))
            X_test = pickle.load(open("/home/robbe/DeepLCCS/data/X_test_trainset_all_features_onehot.pickle", "rb"))
            global_feats_test = pickle.load(open("/home/robbe/DeepLCCS/data/global_feats_test_trainset_all_features_onehot.pickle", "rb"))
            ccs_df_train = pickle.load(open("/home/robbe/DeepLCCS/data/ccs_df_train_trainset_all_features_onehot.pickle", "rb"))
            ccs_df_test = pickle.load(open("/home/robbe/DeepLCCS/data/ccs_df_test_trainset_all_features_onehot.pickle", "rb"))
            logger.info("Training and test split already exist, loading them...")

    except IOError:
        logger.info("Training and test split do not exist, creating them...")
        (
            X_train,
            global_feats_train,
            X_test,
            global_feats_test,
            ccs_df_train,
            ccs_df_test,
        ) = get_features(ccs_df, dataset, architecture, num_lstm, info)

        if dataset == "full":
            pickle.dump(X_train, open("/home/robbe/DeepLCCS/data/X_train_full-trainsetwithlength.pickle", "wb"))
            pickle.dump(
                global_feats_train, open("/home/robbe/DeepLCCS/data/global_feats_train_full-trainsetwithlength.pickle", "wb")
            )
            pickle.dump(X_test, open("/home/robbe/DeepLCCS/data/X_test_full-trainsetwithlength.pickle", "wb"))
            pickle.dump(global_feats_test, open("/home/robbe/DeepLCCS/data/global_feats_test_full-trainsetwithlength.pickle", "wb"))
            pickle.dump(ccs_df_train, open("/home/robbe/DeepLCCS/data/ccs_df_train_full-trainsetwithlength.pickle", "wb"))
            pickle.dump(ccs_df_test, open("/home/robbe/DeepLCCS/data/ccs_df_test_full-trainsetwithlength.pickle", "wb"))

        elif dataset == "sample":
            pickle.dump(X_train, open("/home/robbe/DeepLCCS/data/X_train_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "wb"))
            pickle.dump(
                global_feats_train, open("/home/robbe/DeepLCCS/data/global_feats_train_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "wb")
            )
            pickle.dump(X_test, open("/home/robbe/DeepLCCS/data/X_test_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "wb"))
            pickle.dump(
                global_feats_test, open("/home/robbe/DeepLCCS/data/global_feats_test_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "wb")
            )
            pickle.dump(ccs_df_train, open("/home/robbe/DeepLCCS/data/ccs_df_train_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "wb"))
            pickle.dump(ccs_df_test, open("/home/robbe/DeepLCCS/data/ccs_df_test_sample_samplewithmass_fixed-Hrel-Bulkyrel-Acidrel-KRrel", "wb"))
            logger.info("Training and test split created and saved.")

        else:
            logger.info("Making custom data")
            pickle.dump(X_train, open("/home/robbe/DeepLCCS/data/X_train_trainset_all_features_onehot.pickle", "wb"))
            pickle.dump(global_feats_train, open("/home/robbe/DeepLCCS/data/global_feats_train_trainset_all_features_onehot.pickle", "wb"))
            pickle.dump(X_test, open("/home/robbe/DeepLCCS/data/X_test_trainset_all_features_onehot.pickle", "wb"))
            pickle.dump(global_feats_test, open("/home/robbe/DeepLCCS/data/global_feats_test_trainset_all_features_onehot.pickle", "wb"))
            pickle.dump(ccs_df_train, open("/home/robbe/DeepLCCS/data/ccs_df_train_trainset_all_features_onehot.pickle", "wb"))
            pickle.dump(ccs_df_test, open("/home/robbe/DeepLCCS/data/ccs_df_test_trainset_all_features_onehot.pickle", "wb"))
            logger.info("Training and test split created and saved.")



    return (
        ccs_df,
        X_train,
        global_feats_train,
        X_test,
        global_feats_test,
        ccs_df_train,
        ccs_df_test,
    )
