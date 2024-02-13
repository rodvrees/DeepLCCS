from datetime import datetime
import pickle
import wandb
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Config paramaters
config = {
    ## GENERAL
    "name": "DeepLCCS",
    "time": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
    "save_model": False,

    ## GENERAL MODEL PARAMETERS
    "epochs": 5,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "adam_weight_decay": 0,
    "adam_betas": (0.9, 0.999),
    "loss_function": "mse",
    "v_split": 0.1,

    ## LSTM CHANNEL
    "N_LSTM_layers": 2,
    # "N_LSTM_units": [512, 256, 128],
    "N_LSTM_units1": 512,
    "N_LSTM_units2": 256,
    "N_LSTM_units3": 128,
    "LSTM_dropout": False,
    # "LSTM_dropout_strength": [0, 0],
    "LSTM_dropout_strength1": 0.3,
    "LSTM_dropout_strength2": 0,
    "LSTM_dropout_strength3": 0,
    "LSTM_L1": False,
    "LSTM_L1_strength": [0, 0],
    "LSTM_L2": False,
    "LSTM_L2_strength": [0, 0],
    "LSTM_L1L2": False,

    ## GLOBAL CHANNEL
    "N_global_layers": 4,
    # "N_global_units": [64, 64, 64, 64, 64, 64, 64, 64],
    "N_global_units1": 64,
    "N_global_units2": 2,
    "N_global_units3": 8,
    "N_global_units4": 64,
    "N_global_units5": 16,
    "N_global_units6": 4,
    "N_global_units7": 32,
    "N_global_units8": 64,
    "global_activation": "relu",  # Might want to change this to a list as well to make variable for each layer
    "global_dropout": False,
    # "global_dropout_strength": [0, 0, 0, 0, 0, 0],
    "global_dropout_strength1": 0,
    "global_dropout_strength2": 0,
    "global_dropout_strength3": 0,
    "global_dropout_strength4": 0,
    "global_dropout_strength5": 0.3,
    "global_dropout_strength6": 0,
    "global_dropout_strength7": 0,
    "global_dropout_strength8": 0,
    "global_L1": False,
    "global_L1_strength": [0, 0, 0, 0, 0, 0],
    "global_L2": False,
    "global_L2_strength": [0, 0, 0, 0, 0, 0],
    "global_L1L2": False,

    ## CONCATENATED CHANNEL
    "N_concat_layers": 2,
    # "N_concat_units": [512, 256, 128],
    "N_concat_units1": 512,
    "N_concat_units2": 256,
    "N_concat_units3": 128,
    "N_concat_units4": 64,
    "N_concat_units5": 64,
    "N_concat_units6": 64,
    "N_concat_units7": 64,
    "N_concat_units8": 64,
    "concat_activation": "relu",
    "concat_dropout": False,
    # "concat_dropout_strength": [0, 0, 0],
    "concat_dropout_strength1": 0.3,
    "concat_dropout_strength2": 0,
    "concat_dropout_strength3": 0,
    "concat_dropout_strength4": 0,
    "concat_dropout_strength5": 0,
    "concat_dropout_strength6": 0,
    "concat_dropout_strength7": 0,
    "concat_dropout_strength8": 0,
    "concat_L1": False,
    "concat_L1_strength": [0, 0, 0],
    "concat_L2": False,
    "concat_L2_strength": [0, 0, 0],
}

# Set-up WandB
wandb_run = wandb.init(
    project="pytorch_DeepLCCS_testing",
    name=config["name"] + "-" + config["time"],
    save_code=False,
    config=config,
)

config = wandb.config


def get_loss_function(loss_function_name):
    if loss_function_name == "mse":
        return nn.MSELoss()


def get_optimizer(optimizer_name, model, learning_rate, betas, weight_decay):
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
        )


def mean_absolute_error(targets, predictions):
    """
    Calculate the mean absolute error (MAE).
    """
    return np.mean(np.abs(targets - predictions))


def train_model(
    model, criterion, optimizer, train_loader, valid_loader, num_epochs=100
):
    mae_values = []
    val_mae_values = []
    best_loss = np.Inf
    best_mae = np.Inf
    best_val_loss = np.Inf
    best_val_mae = np.Inf

    for epoch in range(num_epochs):
        start_time = datetime.now()
        model.train()  # Set the model to training mode

        for inputs1_batch, inputs2_batch, targets_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs1_batch, inputs2_batch)

            # Compute the loss
            loss = criterion(outputs, targets_batch)

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
            for inputs1_batch, inputs2_batch, targets_batch in train_loader:
                outputs = model(inputs1_batch, inputs2_batch)
                train_predictions.extend(outputs.cpu().numpy())
                train_targets.extend(targets_batch.cpu().numpy())

            for inputs1_batch, inputs2_batch, targets_batch in valid_loader:
                outputs = model(inputs1_batch, inputs2_batch)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets_batch.cpu().numpy())

            mae = mean_absolute_error(
                np.array(train_targets), np.array(train_predictions)
            )
            mae_values.append(mae)
            val_mae = mean_absolute_error(
                np.array(val_targets), np.array(val_predictions)
            )
            val_mae_values.append(val_mae)

        if loss < best_loss:
            best_loss = loss
        if mae < best_mae:
            best_mae = mae
        if val_mae < best_val_mae:
            best_val_mae = val_mae
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            if config["save_model"]:
                torch.save(model.state_dict(), '/home/robbe/DeepLCCS/models/{}_model.pth'.format(config["name"] + "-" + config["time"]))

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
            f"Epoch [{epoch+1}/{num_epochs}]: Loss: {loss:.4f}, MAE: {mae:.4f}, Validation Loss: {validation_loss:.4f}, Validation MAE: {val_mae:.4f}, Training time: {train_time.total_seconds()} seconds"
        )

    print("Training finished!")


def validate_model(model, criterion, valid_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for inputs1_batch, inputs2_batch, targets_batch in valid_loader:
            # Forward pass
            outputs = model(inputs1_batch, inputs2_batch)

            # Compute the loss
            loss = criterion(outputs, targets_batch)

            total_loss += loss.item()

    # Calculate average validation loss
    avg_loss = total_loss / len(valid_loader)

    return avg_loss


def evaluate_model(model, test_loader, config):
    model.eval()  # Set the model to evaluation mode
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs1_batch, inputs2_batch, targets_batch in test_loader:
            # Forward pass
            outputs = model(inputs1_batch, inputs2_batch)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(targets_batch.cpu().numpy())

    mae = mean_absolute_error(np.array(test_targets), np.array(test_predictions))
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

# Model
class IM2DeepModel(nn.Module):
    def __init__(self, config):
        super(IM2DeepModel, self).__init__()
        self.config = config

        # Bidirectional LSTM layer
        self.lstms = nn.ModuleList()
        self.lstms.append(
            nn.LSTM(6, self.config['N_LSTM_units1'], batch_first=True, bidirectional=True, dropout=self.config['LSTM_dropout_strength1'] if self.config['LSTM_dropout'] else 0)
        )
        for i in range(1, self.config['N_LSTM_layers']):
            self.lstms.append(
                nn.LSTM(self.config['N_LSTM_units{}'.format(i)] * 2, self.config['N_LSTM_units{}'.format(i+1)], batch_first=True, bidirectional=True, dropout=self.config['LSTM_dropout_strength{}'.format(i+1)] if self.config['LSTM_dropout'] else 0)
            )

        # Dense network for input 2
        global_dense_layers = []
        global_dense_layers.append(nn.Linear(8, self.config["N_global_units1"]))
        global_dense_layers.append(
            self._get_activation(config["global_activation"])
        )  # Define above

        for l in range(1, self.config["N_global_layers"]):
            global_dense_layers.append(
                nn.Linear(
                    self.config["N_global_units{}".format(l)],
                    self.config["N_global_units{}".format(l+1)],

                )
            )
            global_dense_layers.append(
                self._get_activation(config["global_activation"])
            )

        self.global_dense = nn.Sequential(*global_dense_layers)

        # Concatenated size for the fully connected layer
        concat_size = (
            2 * self.config["N_LSTM_units{}".format(self.config["N_LSTM_layers"])]
            + self.config["N_global_units{}".format(self.config['N_global_layers'])]
        )

        # Fully connected layer
        concat_dense_layers = []
        concat_dense_layers.append(
            nn.Linear(concat_size, self.config["N_concat_units1"])
        )
        concat_dense_layers.append(
            self._get_activation(config["concat_activation"])
        )

        for l in range(1, self.config["N_concat_layers"]):
            concat_dense_layers.append(
                nn.Linear(
                    self.config["N_concat_units{}".format(l)],
                    self.config["N_concat_units{}".format(l+1)],
                )
            )
            concat_dense_layers.append(
                self._get_activation(config["concat_activation"])
            )

        concat_dense_layers.append(nn.Linear(self.config["N_concat_units{}".format(self.config['N_concat_layers'])], 1))
        concat_dense_layers.append(
            self._get_activation(config["concat_activation"])
        )

        self.fc = nn.Sequential(*concat_dense_layers)

    def _get_activation(self, activation_name):
        if activation_name == "relu":
            return nn.ReLU()

    def forward(self, input1, input2):
        # Input 1 through Bidirectional LSTM
        lstm_output = input1
        for lstm in self.lstms:
            lstm_output, _ = lstm(lstm_output)

        # Get the last output of the LSTM
        lstm_output = lstm_output[:, -1, :]

        # Input 2 through Dense Network
        dense_output = self.global_dense(input2)

        # Concatenate the outputs
        concatenated = torch.cat((lstm_output, dense_output), dim=1)

        # Fully connected layer
        output = self.fc(concatenated)

        return output

def main(config):
    # Get data
    ccs_df = pd.read_csv("/home/robbe/DeepLCCS/data/trainset.csv")
    X_train = pickle.load(open("/home/robbe/DeepLCCS/data/X_train_full-onlyDeepLCtrainset.pickle", "rb"))
    global_feats_train = pickle.load(
        open("/home/robbe/DeepLCCS/data/global_feats_train_full-onlyDeepLCtrainset.pickle", "rb")
    )
    X_test = pickle.load(open("/home/robbe/DeepLCCS/data/X_test_full-onlyDeepLCtrainset.pickle", "rb"))
    global_feats_test = pickle.load(
        open("/home/robbe/DeepLCCS/data/global_feats_test_full-onlyDeepLCtrainset.pickle", "rb")
    )
    ccs_df_train = pickle.load(
        open("/home/robbe/DeepLCCS/data/ccs_df_train_full-onlyDeepLCtrainset.pickle", "rb")
    )
    ccs_df_test = pickle.load(
        open("/home/robbe/DeepLCCS/data/ccs_df_test_full-onlyDeepLCtrainset.pickle", "rb")
    )
    X_train_reshaped = X_train.transpose(0, 2, 1)
    X_test_reshaped = X_test.transpose(0, 2, 1)
    y_train = ccs_df_train.loc[:, "tr"]
    y_test = ccs_df_test.loc[:, "tr"]
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)

    # Set-up GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # call model
    model = IM2DeepModel(config)
    print(model)
    model.to(device)

    wandb.log({"Total parameters": sum(p.numel() for p in model.parameters())})

    criterion = get_loss_function(config["loss_function"])
    optimizer = get_optimizer(
        config["optimizer"],
        model=model,
        learning_rate=config["learning_rate"],
        betas=config["adam_betas"],
        weight_decay=config["adam_weight_decay"],
    )

    X_train_input1 = torch.tensor(X_train_reshaped, dtype=torch.float32)
    X_train_input2 = torch.tensor(
        global_feats_train, dtype=torch.float32
    )  # Shape: (num_samples, input2_size)
    X_test_input1 = torch.tensor(X_test_reshaped, dtype=torch.float32)
    X_test_input2 = torch.tensor(
        global_feats_test, dtype=torch.float32
    )  # Shape: (num_samples, input2_size)
    y_train = torch.tensor(
        y_train_reshaped, dtype=torch.float32
    )  # Shape: (num_samples, output_size)
    y_test = torch.tensor(
        y_test_reshaped, dtype=torch.float32
    )  # Shape: (num_samples, output_size)
    X_train_input1 = X_train_input1.to(device)
    X_train_input2 = X_train_input2.to(device)
    y_train = y_train.to(device)
    X_test_input1 = X_test_input1.to(device)
    X_test_input2 = X_test_input2.to(device)
    y_test = y_test.to(device)
    # Split the data into training and validation sets
    X_train_input1, X_val_input1, X_train_input2, X_val_input2, y_train, y_val = (
        train_test_split(
            X_train_input1,
            X_train_input2,
            y_train,
            test_size=config["v_split"],
            random_state=42,
        )
    )
    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(X_train_input1, X_train_input2, y_train)
    val_dataset = TensorDataset(X_val_input1, X_val_input2, y_val)
    test_dataset = TensorDataset(X_test_input1, X_test_input2, y_test)
    # Define DataLoader for training and validation sets
    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=config["epochs"],
    )
    mae, r, perc_95, test_df = evaluate_model(
        model, test_loader, config
    )

    wandb.finish()

    plot_predictions(test_df, config, mae, r, perc_95)


if __name__ == "__main__":
    main(config)
