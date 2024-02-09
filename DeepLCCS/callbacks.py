import wandb
from tensorflow.keras.callbacks import Callback

class BestLossTracker(Callback):
    def __init__(self):
        super(BestLossTracker, self).__init__()
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_mean_absolute_error = float('inf')
        self.best_val_mean_absolute_error = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        current_val_loss = logs.get('val_loss')
        current_mean_absolute_error = logs.get('mean_absolute_error')
        current_val_mean_absolute_error = logs.get('val_mean_absolute_error')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            wandb.log({'best_loss': self.best_loss}, commit=False)
        # else:
        #     wandb.log({'best_loss': self.best_loss}, commit=False)
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            wandb.log({'best_val_loss': self.best_val_loss}, commit=False)
        # else:
        #     wandb.log({'best_val_loss': self.best_val_loss}, commit=False)
        if current_mean_absolute_error < self.best_mean_absolute_error:
            self.best_mean_absolute_error = current_mean_absolute_error
            wandb.log({'best_mean_absolute_error': self.best_mean_absolute_error}, commit=False)
        # else:
        #     wandb.log({'best_mean_absolute_error': self.best_mean_absolute_error}, commit=False)
        if current_val_mean_absolute_error < self.best_val_mean_absolute_error:
            self.best_val_mean_absolute_error = current_val_mean_absolute_error
            wandb.log({'best_val_mean_absolute_error': self.best_val_mean_absolute_error}, commit=False)
        # else:
        #     wandb.log({'best_val_mean_absolute_error': self.best_val_mean_absolute_error}, commit=False)


