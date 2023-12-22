from DeepLCCS._argument_parser import parse_args
from DeepLCCS.data_extractor import get_data
from DeepLCCS.model import compile_model, fit_model
from DeepLCCS.plot_results import plot_pred_results
from DeepLCCS.wandb_setup import start_wandb, stop_wandb


def main():
    dataset, args = parse_args()
    (
        ccs_df,
        X_train,
        global_feats_train,
        X_test,
        global_feats_test,
        ccs_df_train,
        ccs_df_test,
    ) = get_data(dataset, args=args)

    config = start_wandb(args)

    model = compile_model(config, X_train)
    fit_model(
        config.architecture, model, X_train, global_feats_train, ccs_df_train, config
    )

    stop_wandb()

    # Predict CCS values test set
    ccs_df_test["LSTM_predictions"] = model.predict((X_test, global_feats_test))
    ccs_df_test.to_csv(
        "./preds/{}_{}_{}_{}.csv".format(
            args.dataset, args.architecture, args.num_lstm, args.info
        )
    )

    plot_pred_results(ccs_df, ccs_df_test, ccs_df_train, args=args)


if __name__ == "__main__":
    main()
