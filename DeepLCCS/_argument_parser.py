import argparse


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
    parser.add_argument("--num_lstm", type=int, default=24, help="Number of LSTM units")
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
    parser.add_argument("--DEBUG", type=bool, default=False, help="Debug mode")
    parser.add_argument(
        "--kernel_size", type=int, default=10, help="Kernel size for CNN"
    )
    parser.add_argument("--strides", type=int, default=1, help="Strides for CNN")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_arguments(
        "--num_dense_layers", type=int, default=1, help="Number of dense layers"
    )
    parser.add_arguments(
        "--num_lstm_layers", type=int, default=1, help="Number of LSTM layers"
    )
    args = parser.parse_args()

    dataset = args.dataset

    return dataset, args
