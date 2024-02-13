import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback
from DeepLCCS.callbacks import BestLossTracker
import datetime
import logging
import wandb

best_loss_tracker = BestLossTracker()

logger = logging.getLogger(__name__)

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
    def __init__(self, X_train, num_lstm, dropout, regularizer=None, l1_strength=0, l2_strength=0):
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
                return_sequences=True,
                recurrent_dropout=dropout,
                # kernel_regularizer=tf.keras.regularizers.l2(0.05)
            )
        )(self.input_layer)
        self.lstm_layer2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                int(num_lstm/2),
                return_sequences=False,
                recurrent_dropout=dropout,
                # kernel_regularizer=tf.keras.regularizers.l2(0.05)
            )
        )(self.lstm_layer)


        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.lstm_layer2)

class GlobalFeaturesDense(tf.keras.Model):
    def __init__(self, num_C_dense, activation, regularizer=None, l1_strength=0, l2_strength=0):
        super(GlobalFeaturesDense, self).__init__()
        self.num_C_dense = num_C_dense
        self.activation = activation
        self.regularizer = regularizer
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

        self.input_layer = tf.keras.Input(shape=(8,))
        self.dense_layer1 = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            # kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.input_layer)
        # self.dropout_layer = tf.keras.layers.Dropout(0.2)(self.dense_layer1)
        self.dense_layer2 = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            # kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.dense_layer1)
        self.dense_layer3 = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            # kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.dense_layer2)
        self.dense_layer4 = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            # kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.dense_layer3)
        self.dense_layer5 = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            # kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.dense_layer4)
        self.dense_layer6 = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            # kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.dense_layer5)
        self.dense_layer7 = tf.keras.layers.Dense(
            num_C_dense,
            activation=activation,
            # kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(self.dense_layer6)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.dense_layer7)

class FinalModel(tf.keras.Model):
    def __init__(self, bidirectional_lstm, global_features_dense, num_concat_dense, activation, loss, learning_rate, regularizer=None, l1_strength=0, l2_strength=0):
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
            self.num_concat_dense[0], activation=activation, kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(concatenated)
        concatenated = tf.keras.layers.Dense(
            self.num_concat_dense[1], activation=activation, kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(concatenated)
        concatenated = tf.keras.layers.Dense(
            self.num_concat_dense[2], activation=activation, kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(concatenated)
        concatenated = tf.keras.layers.Dense(
            self.num_concat_dense[3], activation=activation, kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
        )(concatenated)
        concatenated = tf.keras.layers.Dense(1, activation=activation,
                                            #  kernel_regularizer=regularizer_setup(regularizer, l1_strength, l2_strength)
                                             )(concatenated)


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
    final_model = FinalModel(bidirectional_lstm, global_features_dense, config.num_concat_dense, config.activation, config.loss, config.learning_rate, config.regularizer, config.regularizer_strength_l1, config.regularizer_strength_l2)
    final_model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss=config.loss, metrics=config.metrics)
    return final_model

def fit_model(model, X_train, global_feats_train, ccs_df_train, config, time):
    #TODO: hardcoded path for now should be in config and also shouldn't rely on config.info
    # mcp_save = ModelCheckpoint(filepath = '/home/robbe/DeepLCCS/models/{}.weights.keras'.format(config.info, time), save_best_only=True, save_weights_only=True, monitor='val_mean_absolute_error', mode='min', save_format='tf')
    history = model.fit(
            (X_train, global_feats_train),
            ccs_df_train.loc[:, "tr"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.v_split,
            callbacks=[WandbMetricsLogger(), best_loss_tracker],
            # shuffle=True
            )
    wandb.log({'model_params': model.count_params()})
    logger.info(model.summary())
    #Save best model
    # best_model.save('/home/robbe/DeepLCCS/models/{}.h5'.format(config.info), save_format='h5')
    return history
