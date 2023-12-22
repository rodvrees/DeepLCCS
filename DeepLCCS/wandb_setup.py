import wandb
def start_wandb(args):
    wandb.init(
            project="DeepLCCS",
            name="{}_{}_{}_{}".format(
                args.dataset, args.architecture, args.num_lstm, args.info
            ),
            save_code=True,
            config={
                "architecture": args.architecture,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_lstm": args.num_lstm,
                "num_C_dense": args.num_C_dense,
                "num_concat_dense": args.num_concat_dense,
                "v_split": args.v_split,
                "optimizer": args.optimizer,
                "loss": args.loss,
                "metrics": args.metrics,
                "activation": args.activation,
                "dataset": args.dataset,
                "dropout_lstm": args.dropout_lstm,
                "dropout_C_dense": args.dropout_C_dense,
                "dropout_concat_dense": args.dropout_concat_dense,
                "info": args.info,
                "DEBUG": args.DEBUG,
                "kernel_size": args.kernel_size,
                "strides": args.strides,
                "learning_rate": args.learning_rate,
                "num_dense_layers": args.num_dense_layers,
                "num_lstm_layers": args.num_lstm_layers,
            },
        )

    config = wandb.config
    return config

def stop_wandb():
    wandb.finish()