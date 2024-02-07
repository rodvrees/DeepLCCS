import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

def regularizer_setup(regularizer, l1_strength, l2_strength):
    if regularizer is None:
        return None
    elif regularizer.lower() == 'l1':
        return tf.keras.regularizers.l1(l1_strength)
    elif regularizer.lower() == 'l2':
        return tf.keras.regularizers.l2(l2_strength)
    elif regularizer.lower() == 'l1_l2':
        return tf.keras.regularizers.l1_l2(l1_strength, l2_strength)
    else:
        raise ValueError("Invalid regularizer option")

class BidirectionalLSTM(tf.keras.Model):
    def __init__(self, X_train, num_lstm, dropout, regularizer=None, l1_strength=0.01, l2_strength=0.01):
        super(BidirectionalLSTM, self).__init__()
        self.X_train = X_train
        self.num_lstm = num_lstm
        self.dropout = dropout
        self.regularizer = regularizer
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

        self.input_layer = tf.keras.Input(shape=(None, X_train.shape[2]))
        self.lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                num_lstm,
                return_sequences=False,
                dropout=dropout,
                kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
            )
        )(self.input_layer)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.lstm_layer)

class GlobalFeaturesDense(tf.keras.Model):
    def __init__(self, num_C_dense, activation, regularizer=None, l1_strength=0.01, l2_strength=0.01):
        super(GlobalFeaturesDense, self).__init__()
        self.num_C_dense = num_C_dense
        self.activation = activation
        self.regularizer = regularizer
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

        self.input_layer = tf.keras.Input(shape=(9,))
        self.dense_layer = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.input_layer)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.dense_layer)

class FinalModel(tf.keras.Model):
    def __init__(self, bidirectional_lstm, global_features_dense, num_concat_dense, activation, loss, learning_rate):
        super(FinalModel, self).__init__()

        self.bidirectional_lstm = bidirectional_lstm
        self.global_features_dense = global_features_dense
        self.num_concat_dense = num_concat_dense
        self.activation = activation
        self.loss = loss
        # self.metrics = metrics
        self.learning_rate = learning_rate

        # Combine the outputs of bidirectional_lstm and global_features_dense
        concatenated = tf.keras.layers.concatenate([self.bidirectional_lstm.model.output, self.global_features_dense.model.output], axis=-1)

        # Dense layers after concatenation
        concatenated = tf.keras.layers.Dense(
            self.num_concat_dense[0], activation=activation
        )(concatenated)
        concatenated = tf.keras.layers.Dense(
            self.num_concat_dense[1], activation=activation
        )(concatenated)
        concatenated = tf.keras.layers.Dense(1, activation=activation)(concatenated)

        self.model = tf.keras.Model(
            inputs=[self.bidirectional_lstm.model.input, self.global_features_dense.model.input],
            outputs=concatenated
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=self.loss,
            metrics=self._metrics
        )

    def call(self, inputs):
        return self.model(inputs)

def compile_model(config, X_train):
    print(type(config.metrics))
    bidirectional_lstm = BidirectionalLSTM(X_train, config.num_lstm, config.dropout_lstm, config.regularizer, config.regularizer_strength_l1, config.regularizer_strength_l2)
    global_features_dense = GlobalFeaturesDense(config.num_C_dense, config.activation, config.regularizer, config.regularizer_strength_l1, config.regularizer_strength_l2)
    final_model = FinalModel(bidirectional_lstm, global_features_dense, config.num_concat_dense, config.activation, config.loss, config.learning_rate)
    final_model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss=config.loss, metrics=config.metrics)
    return final_model

def fit_model(model, X_train, global_feats_train, ccs_df_train, config):
    history = model.fit(
            (X_train, global_feats_train),
            ccs_df_train.loc[:, "tr"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.v_split,
            callbacks=[WandbMetricsLogger(log_freq=5)])
    return history
