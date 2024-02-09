import importlib
from deeplcretrainer import deeplcretrainer
importlib.reload(deeplcretrainer)
import sys
import warnings
warnings.filterwarnings('ignore')
df_train_file = '/home/robbe/DeepLCCS/data/peprec_CCS.csv'

models_subtr = deeplcretrainer.retrain(
    [df_train_file],
    mods_transfer_learning=[],
    freeze_layers=False,
    batch_size=128,
    n_epochs=200,
    ratio_valid=0.90,
    outpath='/home/robbe/DeepLCCS/models/DeepLC_benchmark',
    plot_results=True,
    write_csv_results=True,
    predict_ccs=True
)
