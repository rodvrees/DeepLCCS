import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import deeplcretrainer
import deeplc
from collections import Counter
import os
from scipy.stats import pearsonr
import tensorflow.compat.v1 as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint 
import argparse
# import models_bb.APD_mimic as apd

parser = argparse.ArgumentParser(description='Train a DeepLCCS model.')
parser.add_argument('--dataset', type=str, default='full', help='full, sample or path to csv file')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train the model')
parser.add_argument('--num_lstm', type=int, default=12, help='Number of LSTM units')
parser.add_argument('--num_C_dense', type=int, default=5, help='Number of dense units for charge')
parser.add_argument('--num_concat_dense', type=list, default=[64,32], help='Number of dense units after concatenation')
parser.add_argument('--v_split', type=float, default=0.1, help='Validation split')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--loss', type=str, default='mean_squared_error', help='Loss function to use')  
parser.add_argument('--metrics', type=list, default=['mean_absolute_error'], help='Metrics to use')
parser.add_argument('--activation', type=str, default='relu', help='Activation function to use')
parser.add_argument('--dropout_lstm', type=float, default=0.0, help='Dropout for LSTM')
parser.add_argument('--dropout_C_dense', type=float, default=0.0, help='Dropout for dense layers')
parser.add_argument('--dropout_concat_dense', type=list, default=[0.0,0.0], help='Dropout for dense layers after concatenation')
parser.add_argument('--architecture', type=str, default='LSTM', help='Architecture to use')
parser.add_argument('--info', type=str, default="", help='Extra info to add to the run name')
args = parser.parse_args()

dataset = args.dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if dataset == 'full':
    ccs_df = pd.read_csv("./data/peprec_CCS.csv")
elif dataset == 'sample':
    ccs_df = pd.read_csv('./data/ccs_sample.csv')
else:
    if os.path.isfile(dataset):
        ccs_df = pd.read_csv(dataset)
    else:
        FileNotFoundError(f"File {dataset} not found.")

X_matrix_count = pd.DataFrame(ccs_df["seq"].apply(Counter).to_dict()).fillna(0.0).T
# Get all the index identifiers
all_idx = list(X_matrix_count.index)
random.seed(42)

# Shuffle the index identifiers so we can randomly split them in a testing and training set
random.shuffle(all_idx)

# Select 90 % for training and the remaining 10 % for testing
train_idx = all_idx[0:int(len(all_idx)*0.9)]
test_idx = all_idx[int(len(all_idx)*0.9):]

# Get the train and test indices and point to new variables
ccs_df_train = ccs_df.loc[train_idx,:]
ccs_df_test = ccs_df.loc[test_idx,:]

aa_comp = deeplcretrainer.cnn_functions.read_aa_lib("/home/robbe/CCS_pred/aa_comp_rel.csv")

train_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_train,aa_comp=aa_comp, predict_ccs=True)
test_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_test,aa_comp=aa_comp, predict_ccs=True)

train_df.to_csv("/home/robbe/DeepLCCS/train_df.csv")

wandb.init(project="DeepLCCS", 
           name="{}_{}_{}_{}".format(args.dataset, args.architecture, args.num_lstm, args.info),
           save_code=True,
           config = {'architecture' : args.architecture,
                    'epochs' : args.epochs, 
                    'batch_size' : args.batch_size,
                    'num_lstm' : args.num_lstm,
                    'num_C_dense' : args.num_C_dense,
                    'num_concat_dense' : args.num_concat_dense,
                    'v_split' : args.v_split, 
                    'optimizer' : args.optimizer, 
                    'loss' : args.loss, 
                    'metrics' : args.metrics,
                    'activation' : args.activation,
                    'data_set' : args.dataset,
                    'dropout_lstm' : args.dropout_lstm,
                    'dropout_C_dense' : args.dropout_C_dense,
                    'dropout_concat_dense' : args.dropout_concat_dense})

config = wandb.config

X_train, X_train_sum, X_train_global, X_train_hc, y_train = deeplcretrainer.cnn_functions.get_feat_matrix(train_df)
X_test, X_test_sum, X_test_global, X_test_hc, y_test = deeplcretrainer.cnn_functions.get_feat_matrix(test_df)
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

input_a = tf.keras.Input(shape=(None, X_train.shape[2]))
# Bidirectional LSTM
a = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.num_lstm, return_sequences=True, dropout=config.dropout_lstm))(input_a)
# a = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.num_lstm, return_sequences=True))(a)
a = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.num_lstm, dropout=config.dropout_lstm))(a)
a = tf.keras.Model(inputs=input_a, outputs=a)
# Input for charge
input_b = tf.keras.Input(shape=(1,))
# Dense layers for charge
b = tf.keras.layers.Dense(config.num_C_dense, activation=config.activation)(input_b)
b = tf.keras.Model(inputs=input_b, outputs=b)
# Concatenate the two layers
c = tf.keras.layers.concatenate([a.output, b.output], axis=-1)
# Dense layers after concatenation
c = tf.keras.layers.Dense(config.num_concat_dense[0], activation=config.activation)(c)
c = tf.keras.layers.Dense(config.num_concat_dense[1], activation=config.activation)(c)
c = tf.keras.layers.Dense(1, activation=config.activation)(c)
# Create the final model
model = tf.keras.Model(inputs=[a.input, b.input], outputs=c)
model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)

model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)

# Fit the model on the training data
history = model.fit(
    (X_train, ccs_df_train.loc[:, "charge"]),
    ccs_df_train.loc[:, "tr"],
    epochs=config.epochs,
    batch_size=config.batch_size,
    validation_split=config.v_split,
    callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")]
)

wandb.finish()

# Predict CCS values test set
ccs_df_test["LSTM_predictions"] = model.predict((X_test,ccs_df_test.loc[:,"charge"]))
ccs_df_test.to_csv("/home/robbe/DeepLCCS/preds/{}_{}_{}_{}.csv".format(args.dataset, args.architecture, args.num_lstm, args.info))

if len(ccs_df.index) < 1e4:
    set_alpha = 0.2
    set_size = 3
else:
    set_alpha = 0.05
    set_size = 1

# Scatter plot the observations on the test set against the predictions on the same set
plt.scatter(
    ccs_df_test.loc[ccs_df_test["charge"]==2,"tr"],
    ccs_df_test.loc[ccs_df_test["charge"]==2,"LSTM_predictions"],
    alpha=set_alpha,
    s=set_size,
    label="Z=2"
)

plt.scatter(
    ccs_df_test.loc[ccs_df_test["charge"]==3,"tr"],
    ccs_df_test.loc[ccs_df_test["charge"]==3,"LSTM_predictions"],
    alpha=set_alpha,
    s=set_size,
    label="Z=3"
)

plt.scatter(
    ccs_df_test.loc[ccs_df_test["charge"]==4,"tr"],
    ccs_df_test.loc[ccs_df_test["charge"]==4,"LSTM_predictions"],
    alpha=set_alpha,
    s=set_size,
    label="Z=4"
)

# Plot a diagonal the points should be one
plt.plot([300,1100],[300,1100],c="grey")

legend = plt.legend()

for lh in legend.legendHandles:
    lh.set_sizes([25])
    lh.set_alpha(1)

# Get the predictions and calculate performance metrics
predictions = ccs_df_test["LSTM_predictions"]
mare = round(sum((abs(predictions-ccs_df_test.loc[:,"tr"])/ccs_df_test.loc[:,"tr"])*100)/len(predictions),3)
pcc = round(pearsonr(predictions,ccs_df_test.loc[:,"tr"])[0],3)
perc_95 = round(np.percentile((abs(predictions-ccs_df_test.loc[:,"tr"])/ccs_df_test.loc[:,"tr"])*100,95)*2,2)

plt.title(f"LSTM - PCC: {pcc} - MARE: {mare}% - 95th percentile: {perc_95}%")

ax = plt.gca()
ax.set_aspect('equal')

plt.xlabel("Observed CCS (^2)")
plt.ylabel("Predicted CCS (^2)")
plt.savefig("/home/robbe/DeepLCCS/figs/{}_{}_{}_{}.png".format(args.dataset, args.architecture, args.num_lstm, args.info),dpi=300)
