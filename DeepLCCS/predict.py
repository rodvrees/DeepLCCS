
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_predictions(ccs_df, test_df, args, time):
    if len(ccs_df.index) < 1e4:
        set_alpha = 0.2
        set_size = 3
    else:
        set_alpha = 0.05
        set_size = 1

    # Scatter plot the observations on the test set against the predictions on the same set
    plt.scatter(
        test_df.loc[test_df["charge"] == 2, "tr"],
        test_df.loc[test_df["charge"] == 2, "Predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=2",
    )

    plt.scatter(
        test_df.loc[test_df["charge"] == 3, "tr"],
        test_df.loc[test_df["charge"] == 3, "Predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=3",
    )

    plt.scatter(
        test_df.loc[test_df["charge"] == 4, "tr"],
        test_df.loc[test_df["charge"] == 4, "Predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=4",
    )

    # Plot a diagonal the points should be one
    plt.plot([300, 1100], [300, 1100], c="grey")

    legend = plt.legend()

    for lh in legend.legendHandles:
        lh.set_sizes([25])
        lh.set_alpha(1)

    # Get the predictions and calculate performance metrics
    predictions = test_df["Predictions"]
    mare = round(
        sum(
            (abs(predictions - test_df.loc[:, "tr"]) / test_df.loc[:, "tr"])
            * 100
        )
        / len(predictions),
        3,
    )
    pcc = round(pearsonr(predictions, test_df.loc[:, "tr"])[0], 3)
    perc_95 = round(
        np.percentile(
            (abs(predictions - test_df.loc[:, "tr"]) / test_df.loc[:, "tr"])
            * 100,
            95,
        )
        * 2,
        2,
    )

    plt.title(f"PCC: {pcc} - MARE: {mare}% - 95th percentile: {perc_95}%")

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.xlabel("Observed CCS (^2)")
    plt.ylabel("Predicted CCS (^2)")
    try:
        plt.savefig(
            args.figure_dir + "/{}_{}_{}_{}_{}.png".format(
                args.dataset, args.architecture, args.num_lstm, args.info, time
            ),
            dpi=300,
        )    
    except FileNotFoundError:
        plt.savefig(
            args.figure_dir + "{}_{}_{}_{}_{}.png".format(
                args.dataset, args.architecture, args.num_lstm, args.info, time
            ),
            dpi=300,
        )
        

def predict_and_plot(ccs_df, X_test, global_feats_test, test_df, model, args, time):
    test_df["Predictions"] = model.predict((X_test, global_feats_test))
    test_df.to_csv(
            args.prediction_dir + "/{}_{}_{}_{}_{}.csv".format(
                args.dataset, args.architecture, args.num_lstm, args.info, time
            )
        )
    plot_predictions(ccs_df, test_df, args, time)


