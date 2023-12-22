import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from DeepLCCS.model_bb import (
    CNN_model_first_input,
    CNN_model_second_input,
    CNN_concatenate,
    LSTM_CNN_first_input,
    LSTM_CNN_second_input,
    LSTM_CNN_BiLSTM,
    LSTM_CNN_concatenate,
    LSTM_first_input,
    LSTM_second_input,
    LSTM_concatenate,
)

#TODO APD mimic
def compile_model(config, X_train):
    adam = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    if config.architecture == "CNN":
        a = CNN_model_first_input(
            X_train, config.kernel_size, config.strides, padding="same"
        )
        b = CNN_model_second_input(config.num_C_dense, config.activation)
        model = CNN_concatenate(
            a.input,
            b.input,
            a.output,
            b.output,
            config.num_dense_layers,
            config.activation,
            config.num_concat_dense,
        )
    
    if config.architecture == "CNN+LSTM":
        a = LSTM_CNN_first_input(
            X_train, config.kernel_size, config.strides, padding="same"
        )
        b = LSTM_CNN_second_input(config.num_C_dense, config.activation)
        c = LSTM_CNN_BiLSTM(X_train, config.num_lstm, config.dropout_lstm)
        model = LSTM_CNN_concatenate(
            a.input,
            b.input,
            c.input,
            a.output,
            b.output,
            c.output,
            config.num_dense_layers,
            config.activation,
            config.num_concat_dense,
        )
    
    if config.architecture == "LSTM":
        a = LSTM_first_input(
            X_train, config.num_LSTM_layers, config.num_lstm, config.dropout_lstm
        )
        b = LSTM_second_input(config.num_C_dense, config.num_C_dense, config.activation)
        model = LSTM_concatenate(
            a.input,
            b.input,
            a.output,
            b.output,
            config.num_dense_layers,
            config.activation,
            config.num_concat_dense,
        )
    model.compile(optimizer=adam, loss=config.loss, metrics=config.metrics)    
    return model

def fit_model(model, X_train, global_feats_train, ccs_df_train, config):
    if config.architecture == "CNN+LSMT":
        history = model.fit(
            (X_train, global_feats_train, X_train),
            ccs_df_train.loc[:, "tr"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.v_split,
            callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")])
    else:
        history = model.fit(
            (X_train, global_feats_train),
            ccs_df_train.loc[:, "tr"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.v_split,
            callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")])
    return history