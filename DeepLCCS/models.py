import tensorflow as tf
from tensorflow.keras import layers, models

class CNN_model(models.Model):
    def __init__(self, a, b, c, optimizer, loss, metrics):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
    
    def call(self):