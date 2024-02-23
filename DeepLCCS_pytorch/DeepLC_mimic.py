import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import wandb
import os
import pandas as pd
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt

config = {
    "name": "DeepLC_baseline_kernel2",
    "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "batch_size": 128,
    "learning_rate": 0.001,
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
    "LRelu_negative_slope": 0.1,
    "LRelu_saturation": 20,
    "L1_alpha": 2.5e-7,
}

wandb_run = wandb.init(
    name=config["name"],
    project="DeepLC_hyperparams",
    save_code=False,
    config=config,
)

class LRelu_with_saturation(nn.Module):
    def __init__(self, negative_slope, saturation):
        super(LRelu_with_saturation, self).__init__()
        self.negative_slope = negative_slope
        self.saturation = saturation
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def forward(self, x):
        activated = self.leaky_relu(x)
        return torch.clamp(activated, max=self.saturation)

class DeepLC_mimic(nn.Module):
    def __init__(self):
        super(DeepLC_mimic, self).__init__()
        # self.config = config

        self.ConvAtomComp = nn.ModuleList()
        # AtomComp input size is batch_size x 60 x 6 but should be batch_size x 6 x 60
        self.ConvAtomComp.append(nn.Conv1d(6, 256, 2, padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(0.1, 20))
        self.ConvAtomComp.append(nn.Conv1d(256, 256, 2, padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(0.1, 20))
        self.ConvAtomComp.append(nn.MaxPool1d(2, 2))
        self.ConvAtomComp.append(nn.Conv1d(256, 128, 2, padding='same')) #Input is probably 128 now?
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(0.1, 20))
        self.ConvAtomComp.append(nn.Conv1d(128, 128, 2, padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(0.1, 20))
        self.ConvAtomComp.append(nn.MaxPool1d(2, 2))
        self.ConvAtomComp.append(nn.Conv1d(128, 64, 2, padding='same')) #Input is probably 64 now?
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(0.1, 20))
        self.ConvAtomComp.append(nn.Conv1d(64, 64, 2, padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(0.1, 20))

        # Flatten
        self.ConvAtomComp.append(nn.Flatten())

        self.ConvDiatomComp = nn.ModuleList()
        # DiatomComp input size is batch_size x 30 x 6 but should be batch_size x 6 x 30
        self.ConvDiatomComp.append(nn.Conv1d(6, 128, 2, padding='same'))
        self.ConvDiatomComp.append(nn.LeakyReLU())
        self.ConvDiatomComp.append(nn.Conv1d(128, 128, 2, padding='same'))
        self.ConvDiatomComp.append(nn.LeakyReLU())
        self.ConvDiatomComp.append(nn.MaxPool1d(2, 2))
        self.ConvDiatomComp.append(nn.Conv1d(128, 64, 2, padding='same')) #Input is probably 64 now?
        self.ConvDiatomComp.append(nn.LeakyReLU())
        self.ConvDiatomComp.append(nn.Conv1d(64, 64, 2, padding='same'))
        self.ConvDiatomComp.append(nn.LeakyReLU())
        # Flatten
        self.ConvDiatomComp.append(nn.Flatten())

        self.ConvGlobal = nn.ModuleList()
        # Global input size is batch_size x 60
        self.ConvGlobal.append(nn.Linear(60, 16))
        # self.ConvGlobal.append(nn.LeakyReLU())
        self.ConvGlobal.append(LRelu_with_saturation(0.1, 20))
        self.ConvGlobal.append(nn.Linear(16, 16))
        # self.ConvGlobal.append(nn.LeakyReLU())
        self.ConvGlobal.append(LRelu_with_saturation(0.1, 20))
        self.ConvGlobal.append(nn.Linear(16, 16))
        # self.ConvGlobal.append(nn.LeakyReLU())
        self.ConvGlobal.append(LRelu_with_saturation(0.1, 20))

        # One-hot encoding
        self.OneHot = nn.ModuleList()
        self.OneHot.append(nn.Conv1d(20, 2, 2, padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.Conv1d(2, 2, 2, padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.MaxPool1d(10, 10))
        self.OneHot.append(nn.Flatten())

        # Concatenate
        self.Concat = nn.ModuleList()
        self.Concat.append(nn.Linear(1948, 128))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(0.1, 20))
        self.Concat.append(nn.Linear(128, 128))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(0.1, 20))
        self.Concat.append(nn.Linear(128, 128))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(0.1, 20))
        self.Concat.append(nn.Linear(128, 128))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(0.1, 20))
        self.Concat.append(nn.Linear(128, 128))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(0.1, 20))
        self.Concat.append(nn.Linear(128, 1))

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)
        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)
        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)
        for layer in self.OneHot:
            one_hot = layer(one_hot)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)
        for layer in self.Concat:
            concatenated = layer(concatenated)
        output = concatenated
        return output

def mean_absolute_error(targets, predictions):
    """
    Calculate the mean absolute error (MAE).
    """
    return np.mean(np.abs(targets - predictions))

def train_model(
    model, criterion, optimizer, train_loader, valid_loader, num_epochs=100, device="cuda:1"):
    mae_values = []
    val_mae_values = []
    best_loss = np.Inf
    best_mae = np.Inf
    best_val_loss = np.Inf
    best_val_mae = np.Inf

    for epoch in range(num_epochs):
        start_time = datetime.now()
        model.train()  # Set the model to training mode

        for AtomEnc_batch, Globals_batch, DiAminoAtomEnc_batch, OneHot_batch, targets_batch in train_loader:
            # wandb.log({"current_learning_rate" : scheduler.get_last_lr()[0]})
            optimizer.zero_grad()

            # Forward pass
            outputs = model(AtomEnc_batch, Globals_batch, DiAminoAtomEnc_batch, OneHot_batch)

            # Compute the loss
            loss = criterion(outputs, targets_batch.unsqueeze(1))

            # L1 regularization
            l1_regularization = torch.tensor(0.0, requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name and ('Concat' in name or 'Conv' in name) and 'OneHot' not in name:
                    l1_regularization += torch.norm(param, 1)
            loss += config['L1_alpha'] * l1_regularization


            mae = mean_absolute_error(targets_batch.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            wandb.log({"step/loss": loss})
            wandb.log({"step/mae": mae})
            if (epoch == 5) & (loss > 200000):
                print("Loss exploded")
                raise ValueError("Loss exploded")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Validate the model after each epoch
        validation_loss = validate_model(model, criterion, valid_loader)
        with torch.no_grad():
            train_predictions = []
            train_targets = []
            val_predictions = []
            val_targets = []
            for AtomEnc_batch, DiAminoAtomEnc_batch, Globals_batch, OneHot_batch, targets_batch in train_loader:
                outputs = model(AtomEnc_batch, DiAminoAtomEnc_batch, Globals_batch, OneHot_batch)
                train_predictions.extend(outputs.cpu().numpy())
                train_targets.extend(targets_batch.cpu().numpy())

            for AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val, targets_batch_val in valid_loader:
                outputs = model(AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets_batch_val.cpu().numpy())

            mae = mean_absolute_error(
                np.array(train_targets), np.array(train_predictions).squeeze()
            )
            mae_values.append(mae)

            val_mae = mean_absolute_error(
                np.array(val_targets), np.array(val_predictions).squeeze()
            )
            val_mae_values.append(val_mae)

        if loss < best_loss:
            best_loss = loss
        if mae < best_mae:
            best_mae = mae
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss

        finish_time = datetime.now()
        train_time = finish_time - start_time

        wandb.log({
            "Epoch": epoch + 1,
            "Loss": loss,
            "Mean absolute error": mae,
            "Validation loss": validation_loss,
            "Validation mean absolute error": val_mae,
            "Best Loss": best_loss,
            "Best mean absolute error": best_mae,
            "Best validation loss": best_val_loss,
            "Best validation mean absolute error": best_val_mae,
            "Training time": train_time.total_seconds()
        })
        print(
            f"Epoch [{epoch+1}/{num_epochs}]: Loss: {loss:.4f}, MAE: {mae:.4f}, Validation Loss: {validation_loss:.4f}, Validation MAE: {val_mae:.4f}, Training time: {train_time.total_seconds()} seconds, Learning rate: {optimizer.param_groups[0]['lr']}"
        )

    print("Training finished!")
    return best_model


def validate_model(model, criterion, valid_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val, targets_batch_val in valid_loader:
            # Forward pass
            outputs = model(AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val)

            # Compute the loss
            loss = criterion(outputs, targets_batch_val.unsqueeze(1))

            total_loss += loss.item()

    # Calculate average validation loss
    avg_loss = total_loss / len(valid_loader)

    return avg_loss

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for Test_AtomEnc_batch, Test_DiAminoAtomEnc_batch, Test_Globals_batch, Test_OneHot_batch, y_test_batch in test_loader:
            # Forward pass
            outputs = model(Test_AtomEnc_batch, Test_DiAminoAtomEnc_batch, Test_Globals_batch, Test_OneHot_batch)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(y_test_batch.cpu().numpy())

    mae = mean_absolute_error(np.array(test_targets), np.array(test_predictions).squeeze())
    print(f"Test MAE: {mae:.4f}")
    r, p = stats.pearsonr(
        np.array(test_targets).flatten(), np.array(test_predictions).flatten()
    )
    print(f"Test Pearson R: {r:.4f}")
    perc_95 = round(
        np.percentile(
            (
                abs(
                    np.array(test_predictions).flatten()
                    - np.array(test_targets).flatten()
                )
                / np.array(test_targets).flatten()
            )
            * 100,
            95,
        ),
        2,
    )

    wandb.log({"Test MAE": mae, "Test Pearson R": r, "Test 95th percentile": perc_95})
    # Save the predictions for each sample #TODO: Give this to the df of the samples
    test_df = pd.DataFrame({"Predictions": test_predictions, "Targets": test_targets})
    test_df.to_csv(
        "/home/robbe/DeepLCCS/preds/{}_preds.csv".format(
            config["name"] + "-" + config["time"]
        )
    )
    return mae, r, perc_95, test_df

def plot_predictions(
    test_df, config, mae, r, perc_95
):  # TODO: make charge state a parameter to color
    if len(test_df) < 1e4:
        set_alpha = 0.2
        set_size = 3
    else:
        set_alpha = 0.05
        set_size = 1

    # Plot the predictions
    plt.scatter(test_df["Targets"], test_df["Predictions"], alpha=set_alpha, s=set_size)
    plt.plot([300, 1100], [300, 1100], c="grey")
    plt.xlabel("Observed CCS (^2)")
    plt.ylabel("Predicted CCS (^2)")
    plt.title(f"PCC: {round(r, 4)} - MARE: {round(mae, 4)}% - 95th percentile: {round(perc_95, 4)}%")
    plt.savefig(
        "/home/robbe/DeepLCCS/figs/{}_preds.png".format(
            config["name"] + "-" + config["time"]
        )
    )

def main():
    # Get data
    ccs_df = pd.read_csv("/home/robbe/DeepLCCS/data/trainset.csv")
    Train_AtomEnc = pickle.load(open("/home/robbe/DeepLCCS/data_clean/X_train_AtomEnc-DeepLC.pickle", "rb"))
    Train_Globals = pickle.load(
        open("/home/robbe/DeepLCCS/data_clean/X_train_GlobalFeatures-DeepLC.pickle", "rb")
    )
    Train_DiAminoAtomEnc = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_train_DiAminoAtomEnc-DeepLC.pickle', 'rb'))
    Train_OneHot = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_train_OneHot-DeepLC.pickle', 'rb'))
    y_train = pickle.load(open('/home/robbe/DeepLCCS/data_clean/y_train-DeepLC.pickle', 'rb'))

    Test_AtomEnc = pickle.load(open("/home/robbe/DeepLCCS/data_clean/X_test_AtomEnc-DeepLC.pickle", "rb"))
    Test_Globals = pickle.load(
        open("/home/robbe/DeepLCCS/data_clean/X_test_GlobalFeatures-DeepLC.pickle", "rb")
    )
    Test_DiAminoAtomEnc = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_test_DiAminoAtomEnc-DeepLC.pickle', 'rb'))
    Test_OneHot = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_test_OneHot-DeepLC.pickle', 'rb'))
    y_test = pickle.load(open('/home/robbe/DeepLCCS/data_clean/y_test-DeepLC.pickle', 'rb'))


    # Set-up GPU device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # call model
    model = DeepLC_mimic()
    print(model)
    model.to(device)

    wandb.log({"Total parameters": sum(p.numel() for p in model.parameters())})
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert the data to PyTorch tensors
    Train_AtomEnc = torch.tensor(Train_AtomEnc, dtype=torch.float32).to(device)
    Train_Globals = torch.tensor(Train_Globals, dtype=torch.float32).to(device)
    Train_DiAminoAtomEnc = torch.tensor(Train_DiAminoAtomEnc, dtype=torch.float32).to(device)
    Train_OneHot = torch.tensor(Train_OneHot, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    Test_AtomEnc = torch.tensor(Test_AtomEnc, dtype=torch.float32).to(device)
    Test_Globals = torch.tensor(Test_Globals, dtype=torch.float32).to(device)
    Test_DiAminoAtomEnc = torch.tensor(Test_DiAminoAtomEnc, dtype=torch.float32).to(device)
    Test_OneHot = torch.tensor(Test_OneHot, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Split the data into training and validation sets
    X_train_AtomEnc, X_val_AtomEnc, X_train_Globals, X_val_Globals, X_train_DiAminoAtomEnc, X_val_DiAminoAtomEnc, X_train_OneHot, X_val_OneHot, y_train, y_val = train_test_split(
        Train_AtomEnc, Train_Globals, Train_DiAminoAtomEnc, Train_OneHot, y_train, test_size=0.1, random_state=42)
    # Create data loaders
    train_dataset = TensorDataset(X_train_AtomEnc, X_train_DiAminoAtomEnc, X_train_Globals, X_train_OneHot, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_dataset = TensorDataset(X_val_AtomEnc, X_val_DiAminoAtomEnc, X_val_Globals, X_val_OneHot, y_val)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_dataset = TensorDataset(Test_AtomEnc, Test_DiAminoAtomEnc, Test_Globals, Test_OneHot, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train the model
    best_model = train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=100)
    mae, r, perc_95, test_df = evaluate_model(best_model, test_loader)

    # Plot predictions
    plot_predictions(test_df, config, mae, r, perc_95)

if __name__ == "__main__":
    main()


