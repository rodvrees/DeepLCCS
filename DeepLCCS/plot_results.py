import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_pred_results(ccs_df, ccs_df_test, ccs_df_train, args={}):
    if len(ccs_df.index) < 1e4:
        set_alpha = 0.2
        set_size = 3
    else:
        set_alpha = 0.05
        set_size = 1

    # Scatter plot the observations on the test set against the predictions on the same set
    plt.scatter(
        ccs_df_test.loc[ccs_df_test["charge"] == 2, "tr"],
        ccs_df_test.loc[ccs_df_test["charge"] == 2, "LSTM_predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=2",
    )

    plt.scatter(
        ccs_df_test.loc[ccs_df_test["charge"] == 3, "tr"],
        ccs_df_test.loc[ccs_df_test["charge"] == 3, "LSTM_predictions"],
        alpha=set_alpha,
        s=set_size,
        label="Z=3",
    )

    plt.scatter(
        ccs_df_test.loc[ccs_df_test["charge"] == 4, "tr"],
        ccs_df_test.loc[ccs_df_test["charge"] == 4, "LSTM_predictions"],
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
    predictions = ccs_df_test["LSTM_predictions"]
    mare = round(
        sum(
            (abs(predictions - ccs_df_test.loc[:, "tr"]) / ccs_df_test.loc[:, "tr"])
            * 100
        )
        / len(predictions),
        3,
    )
    pcc = round(pearsonr(predictions, ccs_df_test.loc[:, "tr"])[0], 3)
    perc_95 = round(
        np.percentile(
            (abs(predictions - ccs_df_test.loc[:, "tr"]) / ccs_df_test.loc[:, "tr"])
            * 100,
            95,
        )
        * 2,
        2,
    )

    plt.title(f"LSTM - PCC: {pcc} - MARE: {mare}% - 95th percentile: {perc_95}%")

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.xlabel("Observed CCS (^2)")
    plt.ylabel("Predicted CCS (^2)")
    plt.savefig(
        "./figs/{}_{}_{}_{}.png".format(
            args.dataset, args.architecture, args.num_lstm, args.info
        ),
        dpi=300,
    )