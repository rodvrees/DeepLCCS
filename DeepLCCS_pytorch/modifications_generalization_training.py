from DeepLC_mimic import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import itertools
from collections import Counter
import gc
from deeplcretrainer import cnn_functions
import random
import sys


def contains_mod(mod, contain_str):
    if contain_str in mod:
        return True
    else:
        return False


def get_native_version(index_name):
    return index_name.split("|")[0] + "|"


def delete_mod(modifications, m_to_filter):
    modslist = modifications.split("|")
    indices_to_filter = []
    for i, ele in enumerate(modslist):
        if ele == m_to_filter:
            indices_to_filter.append(i - 1)
            indices_to_filter.append(i)
    for i in sorted(indices_to_filter, reverse=True):
        del modslist[i]
    return "|".join(modslist)


def get_all_mods(mods):
    split_mods = mods.split("|")
    return [split_mods[m] for m in range(1, len(split_mods), 2)]


def train_valid_split(ccs_df, mod, v_split=0.9):
    X_matrix_count = pd.DataFrame(ccs_df["seq"].apply(Counter).to_dict()).fillna(0.0).T
    # Get all the index identifiers
    all_idx = list(X_matrix_count.index)
    random.seed(42)

    # Shuffle the index identifiers so we can randomly split them in a testing and training set
    random.shuffle(all_idx)

    # Select 90 % for training and the remaining 10 % for testing
    train_idx = all_idx[0 : int(len(all_idx) * v_split)]
    valid_idx = all_idx[int(len(all_idx) * v_split) :]

    # Get the train and valid indices and point to new variables
    ccs_df_train = ccs_df.loc[train_idx, :]
    ccs_df_valid = ccs_df.loc[valid_idx, :]
    ccs_df_train.to_csv("ccs_df_train_{}.csv".format(mod))
    ccs_df_valid.to_csv("ccs_df_valid_{}.csv".format(mod))
    return ccs_df_train, ccs_df_valid


def get_features(ccs_df_train, ccs_df_valid, ccs_df_test, ccs_df_test_mod_ignored):
    # ccs_df_train, ccs_df_test = train_test_split(ccs_df)
    train_df = cnn_functions.get_feat_df(ccs_df_train, predict_ccs=True)
    valid_df = cnn_functions.get_feat_df(ccs_df_valid, predict_ccs=True)
    test_df = cnn_functions.get_feat_df(ccs_df_test, predict_ccs=True)
    test_mod_ignored_df = cnn_functions.get_feat_df(
        ccs_df_test_mod_ignored, predict_ccs=True
    )
    train_df["charge"] = ccs_df_train["charge"]
    test_df["charge"] = ccs_df_test["charge"]
    valid_df["charge"] = ccs_df_valid["charge"]
    test_mod_ignored_df["charge"] = ccs_df_test_mod_ignored["charge"]
    train_df["seq"] = ccs_df_train["seq"]
    test_df["seq"] = ccs_df_test["seq"]
    valid_df["seq"] = ccs_df_valid["seq"]
    test_mod_ignored_df["seq"] = ccs_df_test_mod_ignored["seq"]
    train_df["modifications"] = ccs_df_train["modifications"]
    test_df["modifications"] = ccs_df_test["modifications"]
    valid_df["modifications"] = ccs_df_valid["modifications"]
    test_mod_ignored_df["modifications"] = ccs_df_test_mod_ignored["modifications"]

    (
        X_train,
        X_train_sum,
        X_train_global,
        X_train_hc,
        y_train,
    ) = cnn_functions.get_feat_matrix(train_df)

    (
        X_valid,
        X_valid_sum,
        X_valid_global,
        X_valid_hc,
        y_valid,
    ) = cnn_functions.get_feat_matrix(valid_df)

    (
        X_test,
        X_test_sum,
        X_test_global,
        X_test_hc,
        y_test,
    ) = cnn_functions.get_feat_matrix(test_df)

    (
        X_test_mod_ignored,
        X_test_sum_mod_ignored,
        X_test_global_mod_ignored,
        X_test_hc_mod_ignored,
        y_test_mod_ignored,
    ) = cnn_functions.get_feat_matrix(test_mod_ignored_df)

    train_data = {
        "X_train_AtomEnc": X_train,
        "X_train_DiAminoAtomEnc": X_train_sum,
        "X_train_GlobalFeatures": X_train_global,
        "X_train_OneHot": X_train_hc,
        "y_train": y_train,
    }
    test_data = {
        "X_test_AtomEnc": X_test,
        "X_test_DiAminoAtomEnc": X_test_sum,
        "X_test_GlobalFeatures": X_test_global,
        "X_test_OneHot": X_test_hc,
        "y_test": y_test,
    }
    valid_data = {
        "X_valid_AtomEnc": X_valid,
        "X_valid_DiAminoAtomEnc": X_valid_sum,
        "X_valid_GlobalFeatures": X_valid_global,
        "X_valid_OneHot": X_valid_hc,
        "y_valid": y_valid,
    }
    test_data_mod_ignored = {
        "X_test_mod_ignoreed_AtomEnc": X_test_mod_ignored,
        "X_test_mod_ignored_DiAminoAtomEnc": X_test_sum_mod_ignored,
        "X_test_mod_ignored_GlobalFeatures": X_test_global_mod_ignored,
        "X_test_mod_ignored_OneHot": X_test_hc_mod_ignored,
        "y_test_mod_ignored": y_test_mod_ignored,
    }
    return train_data, valid_data, test_data, test_data_mod_ignored


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aa_comp = cnn_functions.read_aa_lib("/home/robbe/DeepLCCS/DeepLCCS/aa_comp_rel.csv")
    df = pd.read_csv("/home/robbe/DeepLCCS/data_clean/peprecMeierAndWillFixedCCS.csv")
    df.fillna("", inplace=True)

    all_pos_mods = list(itertools.chain(*list(df["modifications"].apply(get_all_mods))))
    all_pos_mods = Counter(all_pos_mods)

    for mod in all_pos_mods:
        config = {
            "name": mod,
            "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "batch_size": 128,
            "learning_rate": 0.0001,
            "AtomComp_kernel_size": 2,
            "DiatomComp_kernel_size": 2,
            "One_hot_kernel_size": 2,
            "AtomComp_out_channels_start": 256,
            "DiatomComp_out_channels_start": 128,
            "Global_units": 16,
            "OneHot_out_channels": 2,
            "Concat_units": 128,
            "AtomComp_MaxPool_kernel_size": 2,
            "DiatomComp_MaxPool_kernel_size": 2,
            "OneHot_MaxPool_kernel_size": 10,
            "LRelu_negative_slope": 0.01,
            "LRelu_saturation": 20,
            "L1_alpha": 5e-7,
            'epochs': 100,
        }
        wandb.init(project="DeepLCCS_mod", name="{}_{}".format(config['name'], config["time"]), config=config)
        config = wandb.config

        print("Current mod:", mod)

        gc.collect()
        model_name = "models/modifications/{}.hdf5".format(mod)

        model = DeepLC_mimic(config)
        print(model)
        model.to(device)

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        mod_presence = df["modifications"].apply(lambda x: contains_mod(mod, x))
        df_test = df[mod_presence]
        df_train = df[~mod_presence]
        df_train, df_valid = train_valid_split(df_train, mod, 0.9)
        df_test_mod_ignored = df_test.copy()
        df_test_mod_ignored["modifications"] = df_test_mod_ignored[
            "modifications"
        ].apply(lambda x: delete_mod(x, mod))

        train_data, valid_data, test_data, test_data_mod_ignored = get_features(
            df_train, df_valid, df_test, df_test_mod_ignored
        )

        # Convert data to Pytorch tensors
        Train_AtomEnc = torch.tensor(
            train_data["X_train_AtomEnc"], device=device, dtype=torch.float32
        )
        Train_DiAminoAtomEnc = torch.tensor(
            train_data["X_train_DiAminoAtomEnc"], device=device, dtype=torch.float32
        )
        Train_GlobalFeatures = torch.tensor(
            train_data["X_train_GlobalFeatures"], device=device, dtype=torch.float32
        )
        Train_OneHot = torch.tensor(
            train_data["X_train_OneHot"], device=device, dtype=torch.float32
        )
        Train_y = torch.tensor(
            train_data["y_train"], device=device, dtype=torch.float32
        )

        Valid_AtomEnc = torch.tensor(
            valid_data["X_valid_AtomEnc"], device=device, dtype=torch.float32
        )
        Valid_DiAminoAtomEnc = torch.tensor(
            valid_data["X_valid_DiAminoAtomEnc"], device=device, dtype=torch.float32
        )
        Valid_GlobalFeatures = torch.tensor(
            valid_data["X_valid_GlobalFeatures"], device=device, dtype=torch.float32
        )
        Valid_OneHot = torch.tensor(
            valid_data["X_valid_OneHot"], device=device, dtype=torch.float32
        )
        Valid_y = torch.tensor(
            valid_data["y_valid"], device=device, dtype=torch.float32
        )

        Test_AtomEnc = torch.tensor(
            test_data["X_test_AtomEnc"], device=device, dtype=torch.float32
        )
        Test_DiAminoAtomEnc = torch.tensor(
            test_data["X_test_DiAminoAtomEnc"], device=device, dtype=torch.float32
        )
        Test_GlobalFeatures = torch.tensor(
            test_data["X_test_GlobalFeatures"], device=device, dtype=torch.float32
        )
        Test_OneHot = torch.tensor(
            test_data["X_test_OneHot"], device=device, dtype=torch.float32
        )
        Test_y = torch.tensor(test_data["y_test"], device=device, dtype=torch.float32)

        Test_mod_ignored_AtomEnc = torch.tensor(
            test_data_mod_ignored["X_test_mod_ignoreed_AtomEnc"],
            device=device,
            dtype=torch.float32,
        )
        Test_mod_ignored_DiAminoAtomEnc = torch.tensor(
            test_data_mod_ignored["X_test_mod_ignored_DiAminoAtomEnc"],
            device=device,
            dtype=torch.float32,
        )
        Test_mod_ignored_GlobalFeatures = torch.tensor(
            test_data_mod_ignored["X_test_mod_ignored_GlobalFeatures"],
            device=device,
            dtype=torch.float32,
        )
        Test_mod_ignored_OneHot = torch.tensor(
            test_data_mod_ignored["X_test_mod_ignored_OneHot"],
            device=device,
            dtype=torch.float32,
        )
        Test_mod_ignored_y = torch.tensor(
            test_data_mod_ignored["y_test_mod_ignored"],
            device=device,
            dtype=torch.float32,
        )

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            Train_AtomEnc,
            Train_DiAminoAtomEnc,
            Train_GlobalFeatures,
            Train_OneHot,
            Train_y,
        )
        valid_dataset = torch.utils.data.TensorDataset(
            Valid_AtomEnc,
            Valid_DiAminoAtomEnc,
            Valid_GlobalFeatures,
            Valid_OneHot,
            Valid_y,
        )
        test_dataset = torch.utils.data.TensorDataset(
            Test_AtomEnc, Test_DiAminoAtomEnc, Test_GlobalFeatures, Test_OneHot, Test_y
        )
        test_mod_ignored_dataset = torch.utils.data.TensorDataset(
            Test_mod_ignored_AtomEnc,
            Test_mod_ignored_DiAminoAtomEnc,
            Test_mod_ignored_GlobalFeatures,
            Test_mod_ignored_OneHot,
            Test_mod_ignored_y,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_mod_ignored_loader = torch.utils.data.DataLoader(
            test_mod_ignored_dataset, batch_size=config["batch_size"], shuffle=True
        )

        # Train the model
        best_model = train_model(
            model,
            criterion,
            optimizer,
            train_loader,
            valid_loader,
            num_epochs=config["num_epochs"],
            device=device,
        )
        torch.save(best_model.state_dict(), model_name)
        mae, r, perc_95, test_df = evaluate_model(
            best_model,
            test_loader,
            device,
            info="Test_mod_{}_encoded".format(mod),
            path="/home/robbe/DeepLCCS/DeepLCCS_pytorch/predictions/",
        )
        mae_mod_ignored, r_mod_ignored, perc_95_mod_ignored, test_df_mod_ignored = (
            evaluate_model(
                best_model,
                test_mod_ignored_loader,
                device,
                info="Test_mod_{}_ignored".format(mod),
                path="/home/robbe/DeepLCCS/DeepLCCS_pytorch/predictions/",
            )
        )

        # Plot predictions
        plot_predictions(
            test_df,
            mod,
            mae,
            r,
            perc_95,
            info="Test_mod_{}_encoded".format(mod),
            path="/home/robbe/DeepLCCS/DeepLCCS_pytorch/figures/modifications/",
        )
        plot_predictions(
            test_df_mod_ignored,
            mod,
            mae_mod_ignored,
            r_mod_ignored,
            perc_95_mod_ignored,
            info="Test_mod_{}_ignored".format(mod),
            path="/home/robbe/DeepLCCS/DeepLCCS_pytorch/figures/modifications/",
        )
        print("Done with mod:", mod)


if __name__ == "__main__":
    run()
