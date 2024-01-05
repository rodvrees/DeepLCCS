import tensorflow as tf
from tensorflow.keras import layers, models
import copy


# TODO This probably will not work, the output of each class should be the output, not a model object
class CNN_model_first_input(models.Model):
    def __init__(self, X_train, kernel_size, strides, padding="same"):
        super().__init__()
        self.input = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
        self.conv1 = layers.Conv1D(
            filters=1, kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.flatten = layers.Flatten()

    def call(self):
        input_a = self.input()
        a = self.conv1(input_a)
        a = self.flatten(a)
        return tf.keras.Model(inputs=input_a, outputs=a)


class CNN_model_second_input(models.Model):
    def __init__(self, num_C_dense, activation):
        super().__init__()
        self.input = layers.Input(shape=(19,))
        self.dense = layers.Dense(num_C_dense, activation=activation)

    def call(self):
        input_b = self.input()
        b = self.dense(input_b)
        return tf.keras.Model(inputs=input_b, outputs=b)


class CNN_concatenate(models.Model):
    def __init__(
        self,
        a_input,
        b_input,
        a_output,
        b_output,
        num_dense_layers,
        activation,
        dense_units,
    ):
        super().__init__()
        self.num_dense_layers = num_dense_layers
        self.dense_units = dense_units
        self.activation = activation
        self.concatenate = layers.Concatenate([a_output, b_output], axis=-1)
        self.dense_final = layers.Dense(1, activation=activation)
        self.a_input = a_input
        self.b_input = b_input

        self.build_dense_layers()

    def build_dense_layers(self):
        for i in range(self.num_dense_layers - 1):
            setattr(
                self,
                "dense_{}".format(i),
                layers.Dense(self.dense_units, activation=self.activation),
            )

    def call(self):
        c = self.concatenate()
        for i in range(self.num_dense_layers - 1):
            c = getattr(self, "dense_{}".format(i))(c)
        c = self.dense_final(c)
        return tf.keras.Model(inputs=[self.a_input, self.b_input], outputs=c)


class LSTM_CNN_first_input(models.Model):
    def __init__(self, X_train, kernel_size, strides, padding="same"):
        super().__init__()
        self.input = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
        self.conv1 = layers.Conv1D(
            filters=128, kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.conv2 = layers.Conv1D(
            filters=32, kernel_size=kernel_size, strides=1, padding=padding
        )
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv3 = layers.Conv1D(
            filters=32, kernel_size=5, strides=1, padding=padding
        )
        self.conv4 = layers.Conv1D(
            filters=32, kernel_size=5, strides=1, padding=padding
        )
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        self.flatten = layers.Flatten()

    def call(self):
        input_a = self.input()
        a = self.conv1(input_a)
        a = self.conv2(a)
        a = self.pool1(a)
        a = self.conv3(a)
        a = self.conv4(a)
        a = self.pool2(a)
        a = self.flatten(a)
        return tf.keras.Model(inputs=input_a, outputs=a)


class LSTM_CNN_second_input(models.Model):
    def __init__(self, num_C_dense, activation):
        super().__init__()
        self.input = layers.Input(shape=(19,))
        self.dense = layers.Dense(num_C_dense, activation=activation)

    def call(self):
        input_b = self.input()
        b = self.dense(input_b)
        return tf.keras.Model(inputs=input_b, outputs=b)


class LSTM_CNN_BiLSTM(models.Model):
    def __init__(self, X_train, num_LSTM, dropout=0.0):
        super().__init__()
        self.input = layers.Input(shape=(None, X_train.shape[2]))
        self.lstm = layers.Bidirectional(
            layers.LSTM(num_LSTM, return_sequences=False, dropout=dropout)
        )

    def call(self):
        input_c = self.input()
        c = self.lstm(input_c)
        return tf.keras.Model(inputs=input_c, outputs=c)


class LSTM_CNN_concatenate(models.Model):
    def __init__(
        self,
        a_input,
        b_input,
        c_input,
        a_output,
        b_output,
        c_output,
        num_dense_layers,
        activation,
        dense_units,
    ):
        super().__init__()
        self.num_dense_layers = num_dense_layers
        self.dense_units = dense_units
        self.activation = activation
        self.concatenate = layers.Concatenate([a_output, b_output, c_output], axis=-1)
        self.dense_final = layers.Dense(1, activation=activation)
        self.a_input = a_input
        self.b_input = b_input
        self.c_input = c_input

        self.build_dense_layers()

    def build_dense_layers(self):
        for i in range(self.num_dense_layers):
            setattr(
                self,
                "dense_{}".format(i),
                layers.Dense(self.dense_units, activation=self.activation),
            )

    def call(self):
        d = self.concatenate()
        for i in range(self.num_dense_layers):
            d = getattr(self, "dense_{}".format(i))(d)
        d = self.dense_final(d)
        return tf.keras.Model(
            inputs=[self.a_input, self.b_input, self.c_input], outputs=d
        )


class LSTM_first_input(models.Model):
    def __init__(self, X_train, num_LSTM_layers, num_LSTM, dropout=0.0):
        super().__init__()
        self.num_LSTM_layers = num_LSTM_layers
        self.num_LSTM = num_LSTM
        self.dropout = dropout

        self.inputlayer = layers.Input(shape=(None, X_train.shape[2]))
        self.final_lstm = layers.Bidirectional(
            layers.LSTM(num_LSTM, return_sequences=False, dropout=dropout)
        )
        self.build_LSTM_layers()

    def build_LSTM_layers(self):
        for i in range(self.num_LSTM_layers - 1):
            setattr(
                self,
                "LSTM_{}".format(i),
                layers.Bidirectional(
                    layers.LSTM(
                        self.LSTM_units, return_sequences=True, dropout=self.dropout
                    )
                ),
            )

    def call(self, inputs):
        a = inputs
        input_a = copy.deepcopy(a)
        for i in range(self.num_LSTM_layers - 1):
            a = getattr(self, "LSTM_{}".format(i))(a)
        return a

class LSTM_second_input(models.Model):
    def __init__(self, num_C_dense, num_dense_layers, activation):
        super().__init__()
        self.inputlayer = layers.Input(shape=(19,))
        self.dense = layers.Dense(num_C_dense, activation=activation)
        self.num_dense_layers = num_dense_layers

    def build_dense_layers(self):
        for i in range(self.num_dense_layers - 1):
            setattr(
                self,
                "dense_{}".format(i),
                layers.Bidirectional(
                    layers.dense(
                        self.dense_units, return_sequences=True, dropout=self.dropout
                    )
                ),
            )

    def call(self, inputs):
        input_b = inputs
        b = copy.deepcopy(input_b)
        for i in range(self.num_dense_layers - 1):
            b = getattr(self, "dense_{}".format(i))(b)
        return tf.keras.Model(inputs=input_b, outputs=b)

class LSTM_concatenate(models.Model):
    def __init__(
        self,
        a_input,
        b_input,
        a_output,
        b_output,
        num_dense_layers,
        activation,
        dense_units,
    ):
        super().__init__()
        self.num_dense_layers = num_dense_layers
        self.dense_units = dense_units
        self.activation = activation
        self.concatenate = layers.Concatenate([a_output, b_output], axis=-1)
        self.dense_final = layers.Dense(1, activation=activation)
        self.a_input = a_input
        self.b_input = b_input

        self.build_dense_layers()

    def build_dense_layers(self):
        for i in range(self.num_dense_layers - 1):
            setattr(
                self,
                "dense_{}".format(i),
                layers.Dense(self.dense_units, activation=self.activation),
            )

    def call(self):
        c = self.concatenate()
        for i in range(self.num_dense_layers - 1):
            c = getattr(self, "dense_{}".format(i))(c)
        c = self.dense_final(c)
        return tf.keras.Model(inputs=[self.a_input, self.b_input], outputs=c)
    
