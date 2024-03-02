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
import copy

config = {
    "name": "Sweep",
    "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "batch_size": 128,
    "learning_rate": 0.0001,
    "AtomComp_kernel_size": 8,
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
    "L1_alpha": 5e-6,
    'epochs': 100,
    'delta': 0,
    'device': '1'
}

wandb_run = wandb.init(
    name=config["name"] + "-" + config["time"],
    project="DeepLC_hyperparams",
    save_code=False,
    config=config,
)

config = wandb.config

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
    def __init__(self, config):
        super(DeepLC_mimic, self).__init__()
        # self.config = config
        self.config = config
        self.ConvAtomComp = nn.ModuleList()
        # AtomComp input size is batch_size x 60 x 6 but should be batch_size x 6 x 60
        self.ConvAtomComp.append(nn.Conv1d(6, config['AtomComp_out_channels_start'], config['AtomComp_kernel_size'], padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start'], config['AtomComp_out_channels_start'], config['AtomComp_kernel_size'], padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(config['AtomComp_MaxPool_kernel_size'], config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start'], config['AtomComp_out_channels_start']//2, config['AtomComp_kernel_size'], padding='same')) #Input is probably 256 now?
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//2, config['AtomComp_out_channels_start']//2, config['AtomComp_kernel_size'], padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(config['AtomComp_MaxPool_kernel_size'], config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//2, config['AtomComp_out_channels_start']//4, config['AtomComp_kernel_size'], padding='same')) #Input is probably 128 now?
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//4, config['AtomComp_out_channels_start']//4, config['AtomComp_kernel_size'], padding='same'))
        # self.ConvAtomComp.append(nn.LeakyReLU())
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))

        # Flatten
        self.ConvAtomComp.append(nn.Flatten())

        ConvAtomCompSize = (60 // (2 * config['AtomComp_MaxPool_kernel_size'])) * (config['AtomComp_out_channels_start']//4)
        print(ConvAtomCompSize)

        self.ConvDiatomComp = nn.ModuleList()
        # DiatomComp input size is batch_size x 30 x 6 but should be batch_size x 6 x 30
        self.ConvDiatomComp.append(nn.Conv1d(6, config['DiatomComp_out_channels_start'], config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start'], config['DiatomComp_out_channels_start'], config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.MaxPool1d(config['DiatomComp_MaxPool_kernel_size'], config['DiatomComp_MaxPool_kernel_size']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start'], config['DiatomComp_out_channels_start']//2, config['DiatomComp_kernel_size'], padding='same')) #Input is probably 64 now?
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start']//2, config['DiatomComp_out_channels_start']//2, config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        # Flatten
        self.ConvDiatomComp.append(nn.Flatten())

        # Calculate the output size of the DiatomComp layers
        ConvDiAtomCompSize = (30 // config['DiatomComp_MaxPool_kernel_size']) * (config['DiatomComp_out_channels_start']//2)
        print(ConvDiAtomCompSize)

        self.ConvGlobal = nn.ModuleList()
        # Global input size is batch_size x 60
        self.ConvGlobal.append(nn.Linear(60, config['Global_units']))
        # self.ConvGlobal.append(nn.LeakyReLU())
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(config['Global_units'], config['Global_units']))
        # self.ConvGlobal.append(nn.LeakyReLU())
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(config['Global_units'], config['Global_units']))
        # self.ConvGlobal.append(nn.LeakyReLU())
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))

        # Calculate the output size of the Global layers
        ConvGlobal_output_size = config['Global_units']
        print(ConvGlobal_output_size)

        # One-hot encoding
        self.OneHot = nn.ModuleList()
        self.OneHot.append(nn.Conv1d(20, config['OneHot_out_channels'], config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.Conv1d(config['OneHot_out_channels'], config['OneHot_out_channels'], config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.MaxPool1d(config['OneHot_MaxPool_kernel_size'], config['OneHot_MaxPool_kernel_size']))
        self.OneHot.append(nn.Flatten())

        # Calculate the output size of the OneHot layers
        conv_output_size_OneHot = ((60 // config['OneHot_MaxPool_kernel_size']) * config['OneHot_out_channels'])
        print(conv_output_size_OneHot)

        # Calculate the total input size for the Concat layer
        total_input_size = ConvAtomCompSize + ConvDiAtomCompSize + ConvGlobal_output_size + conv_output_size_OneHot
        print(total_input_size)

        # Concatenate
        self.Concat = nn.ModuleList()
        self.Concat.append(nn.Linear(total_input_size, config['Concat_units']))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        # self.Concat.append(nn.LeakyReLU())
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], 1))

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

    best_model_epochs = []
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

        if validation_loss < best_val_loss:
            if validation_loss < (best_val_loss - (config['delta'] * best_val_loss)):
                print('Saving best model')
                # best_model = copy.deepcopy(model)
                # best_model = model
                best_model_epochs.append(epoch + 1)
                torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'models/{}_{}_best_model.pth'.format(config['name'], config['time']))
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
            "Training time": train_time.total_seconds(),
            "Best model epochs": best_model_epochs,
        })
        print(
            f"Epoch [{epoch+1}/{num_epochs}]: Loss: {loss:.4f}, MAE: {mae:.4f}, Validation Loss: {validation_loss:.4f}, Validation MAE: {val_mae:.4f}, Training time: {train_time.total_seconds()} seconds, Learning rate: {optimizer.param_groups[0]['lr']}"
        )

    print("Training finished!")
    print("Model was saved on these epochs:", best_model_epochs)
    return model


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

def evaluate_model(model, test_loader, info='', path='/home/robbe/DeepLCCS/preds/'):
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
    mre = np.median(
        abs(np.array(test_predictions).flatten() - np.array(test_targets).flatten())
        / np.array(test_targets).flatten()
    )
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
    if info == 'best':
        wandb.log({"Test MAE": mae, "Test Pearson R": r, "Test 95th percentile": perc_95, "Test MRE": mre})
    # Save the predictions for each sample #TODO: Give this to the df of the samples
    test_df = pd.DataFrame({"Predictions": test_predictions, "Targets": test_targets})
    test_df.to_csv(
        path + "{}_preds.csv".format(
            info + "-" + config["name"] + "-" + config["time"]
        )
    )
    return mae, r, perc_95, test_df

def plot_predictions(
    test_df, config, mae, r, perc_95, info='', path='/home/robbe/DeepLCCS/figs/'
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
        path + "{}_preds.png".format(
            info + "-" + config["name"] + "-" + config["time"]
        )
    )

def main():
    # Get data
    ccs_df = pd.read_csv("/home/robbe/DeepLCCS/data/trainset.csv")
    Train_AtomEnc = pickle.load(open("/home/robbe/DeepLCCS/data_clean/X_train_AtomEnc-SmallerVal.pickle", "rb"))
    Train_Globals = pickle.load(
        open("/home/robbe/DeepLCCS/data_clean/X_train_GlobalFeatures-SmallerVal.pickle", "rb")
    )
    Train_DiAminoAtomEnc = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_train_DiAminoAtomEnc-SmallerVal.pickle', 'rb'))
    Train_OneHot = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_train_OneHot-SmallerVal.pickle', 'rb'))
    y_train = pickle.load(open('/home/robbe/DeepLCCS/data_clean/y_train-SmallerVal.pickle', 'rb'))

    Test_AtomEnc = pickle.load(open("/home/robbe/DeepLCCS/data_clean/X_test_AtomEnc-SmallerVal.pickle", "rb"))
    Test_Globals = pickle.load(
        open("/home/robbe/DeepLCCS/data_clean/X_test_GlobalFeatures-SmallerVal.pickle", "rb")
    )
    Test_DiAminoAtomEnc = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_test_DiAminoAtomEnc-SmallerVal.pickle', 'rb'))
    Test_OneHot = pickle.load(open('/home/robbe/DeepLCCS/data_clean/X_test_OneHot-SmallerVal.pickle', 'rb'))
    y_test = pickle.load(open('/home/robbe/DeepLCCS/data_clean/y_test-SmallerVal.pickle', 'rb'))


    # Set-up GPU device
    device = "cuda:{}".format(config['device']) if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # call model
    model = DeepLC_mimic(config)
    print(model)
    model.to(device)

    wandb.log({"Total parameters": sum(p.numel() for p in model.parameters())})
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

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
        Train_AtomEnc, Train_Globals, Train_DiAminoAtomEnc, Train_OneHot, y_train, test_size=0.01, random_state=42)

    print(X_train_AtomEnc.shape)
    print(X_val_AtomEnc.shape)
    # Create data loaders
    train_dataset = TensorDataset(X_train_AtomEnc, X_train_DiAminoAtomEnc, X_train_Globals, X_train_OneHot, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_dataset = TensorDataset(X_val_AtomEnc, X_val_DiAminoAtomEnc, X_val_Globals, X_val_OneHot, y_val)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataset = TensorDataset(Test_AtomEnc, Test_DiAminoAtomEnc, Test_Globals, Test_OneHot, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Train the model
    model = train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=config['epochs'], device=device)
    # mae, r, perc_95, test_df = evaluate_model(model, test_loader)

    best_model_state_dict = torch.load('models/{}_{}_best_model.pth'.format(config['name'], config['time']))['model_state_dict']
    best_model = DeepLC_mimic(config).to(device)
    best_model.load_state_dict(best_model_state_dict)
    mae_best, r_best, perc_95_best, test_df_best = evaluate_model(best_model, test_loader, info='best')

    # Plot predictions
    plot_predictions(test_df_best, config, mae_best, r_best, perc_95_best)

if __name__ == "__main__":
    main()


