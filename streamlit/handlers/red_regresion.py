# Aca va una red neuronal para regresion
# armar un pipeline con joblib para despues exportarlo a main
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class regresion_pipeline(BaseEstimator, TransformerMixin):
    def __init__(self):







# Define parameters
params = {
    'num_layers': 3,
    'n_units_input': 54,
    'n_units_layer_0': 5,
    'dropout_rate_layer_0': 0.05145265556334144,
    'n_units_layer_1': 75,
    'dropout_rate_layer_1': 0.31260885803548927,
    'n_units_layer_2': 110,
    'dropout_rate_layer_2': 0.33192966823747483,
    'learning_rate': 0.0006470954272360495
}

# Initialize the model
model = Sequential()

# Add input layer
model.add(Dense(units=params['n_units_input'], activation='relu', input_dim=params['n_units_input']))

# Add hidden layers
for i in range(params['num_layers']):
    units = params[f'n_units_layer_{i}']
    dropout_rate = params[f'dropout_rate_layer_{i}']
    
    model.add(Dense(units=units, activation='relu'))
    if dropout_rate > 0:
        model.add(Dropout(rate=dropout_rate))

# Add output layer
model.add(Dense(units=1, activation='sigmoid'))  # Assuming binary classification (1 output unit)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
              loss='binary_crossentropy',  # Assuming binary crossentropy for binary classification
              metrics=['accuracy'])

# Print a summary of the model
model.summary()












#Best parameters: {'num_layers': 3, 'n_units_input': 54,
#  'n_units_layer_0': 5, 'dropout_rate_layer_0': 0.05145265556334144,
#  'n_units_layer_1': 75, 'dropout_rate_layer_1': 0.31260885803548927,
#  'n_units_layer_2': 110, 'dropout_rate_layer_2': 0.33192966823747483,
#  'learning_rate': 0.0006470954272360495}