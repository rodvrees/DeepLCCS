__author__ = ["Robbe Devreese, Robbin Bouwmeester"]
__credits__ = ["Robbe Devreese, Robbin Bouwmeester", "Prof. Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbe Devreese", "Robbin Bouwmeester"]
__email__ = ["robbe.devreese@ugent.be", "robbin.bouwmeester@ugent.be"]

# Import standard modules
import os
import logging
import sys
sys.path.append("/home/robbe/DeepLCCS")
import warnings
import datetime

# Import local modules
# try:
from DeepLCCS.data_extractor import get_data
# from DeepLCCS.model import compile_model, fit_model
from DeepLCCS.LSTMtest import compile_model, fit_model
from DeepLCCS.wandb_setup import start_wandb, stop_wandb
from DeepLCCS._argument_parser import parse_args
from DeepLCCS._exceptions import DeepLCCSException
from DeepLCCS.wandb_setup import start_wandb, stop_wandb
from DeepLCCS.predict import predict_and_plot
from DeepLCCS import __version__

logger = logging.getLogger(__name__)


def setup_logging(passed_level):
    log_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if passed_level.lower() not in log_mapping:
        raise ValueError(
            "Invalid log level. Should be one of the following: ",
            ", ".join(log_mapping.keys()),
        )

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_mapping[passed_level.lower()],
    )


def main():
    "Main function for DeepLCCS"
    args = parse_args()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    setup_logging(args.log_level)
    if args.log_level == "debug":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        logging.getLogger("tensorflow").setLevel(logging.DEBUG)
        warnings.filterwarnings("default", category=DeprecationWarning)
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=UserWarning)
    else:
        os.environ['KMP_WARNINGS'] = '0'

    if args.wandb:
        config = start_wandb(args, now)

    try:

        run(args, now)
    except DeepLCCSException as e:
        logger.exception(e)
        sys.exit(1)


def run(args, time
):
    """Run DeepLCCS training and validation"""
    logger.info("Starting DeepLCCS-trainer version {}".format(__version__))

    (
        ccs_df,
        X_train,
        global_feats_train,
        X_test,
        global_feats_test,
        ccs_df_train,
        ccs_df_test,
    ) = get_data(args.dataset, args.log_level, args.architecture, args.num_lstm, args.info)

    logger.info('Training model...')
    model = compile_model(args, X_train)
    history = fit_model(
        model, X_train, global_feats_train, ccs_df_train, args, time
    )

    stop_wandb()

    # Predict CCS values test set
    logger.info('Predicting and plotting CCS values for test set...')
    predict_and_plot(ccs_df, X_test, global_feats_test, ccs_df_test, model, args, time)

if __name__ == "__main__":
    main()




