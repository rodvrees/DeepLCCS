import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog="DeepLCCS",
        description="Collisional cross section prediction using deep learning",
    )
    io_args = parser.add_argument_group("Input and output files")
    io_args.add_argument(
        "--dataset", type=str, default="sample", help="full, sample or path to csv file"
    )

    model_args = parser.add_argument_group("Model parameters")
    model_args.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train the model"
    )
    model_args.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to train the model"
    )
    model_args.add_argument(
        "--num_lstm", type=int, default=24, help="Number of LSTM units"
    )
    model_args.add_argument(
        "--num_C_dense", type=int, default=5, help="Number of dense units for charge"
    )
    model_args.add_argument(
        "--num_concat_dense",
        type=list,
        default=[64, 32],
        help="Number of dense units after concatenation",
    )
    model_args.add_argument(
        "--v_split", type=float, default=0.1, help="Validation split"
    )
    model_args.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use"
    )
    model_args.add_argument(
        "--loss", type=str, default="mean_squared_error", help="Loss function to use"
    )
    model_args.add_argument(
        "--metrics", type=list, default=["mean_absolute_error"], help="Metrics to use"
    )
    model_args.add_argument(
        "--activation", type=str, default="relu", help="Activation function to use"
    )
    model_args.add_argument(
        "--dropout_lstm", type=float, default=0.0, help="Dropout for LSTM"
    )
    model_args.add_argument(
        "--dropout_C_dense", type=float, default=0.0, help="Dropout for dense layers"
    )
    model_args.add_argument(
        "--dropout_concat_dense",
        type=list,
        default=[0.0, 0.0],
        help="Dropout for dense layers after concatenation",
    )
    model_args.add_argument(
        "--architecture", type=str, default="LSTM", help="Architecture to use"
    )
    model_args.add_argument(
        "--kernel_size", type=int, default=10, help="Kernel size for CNN"
    )
    model_args.add_argument("--strides", type=int, default=1, help="Strides for CNN")
    model_args.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    model_args.add_argument(
        "--num_dense_layers", type=int, default=1, help="Number of dense layers"
    )
    model_args.add_argument(
        "--num_lstm_layers", type=int, default=1, help="Number of LSTM layers"
    )
    other_args = parser.add_argument_group("Other parameters")
    other_args.add_argument(
        "--info", type=str, default="", help="Extra info to add to the run name"
    )
    other_args.add_argument(
        "--log_level", type=str, default="info", help="Logging level to use"
    )
    args = parser.parse_args()

    return args
