'''
Recurrent Autoencoder 

[Encoder] Sequence to Vector RNN -> [Decoder] Vector to Sequence RNN
'''

import keras
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.core import RepeatVector
from tensorflow.python.keras.layers.wrappers import TimeDistributed

# Build the encoding layer
recurrent_encoder = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences = True, input_shape = [None, 28],
    keras.layers.LSTM(30)
    ])

# Build the decoding layer
recurrent_decoder = keras.models.Sequential([
    keras.layers.RepeatVector(28, input_shape = [30]),
    keras.layers.LSTM(100, return_sequences = True),
    keras.layers.TimeDistributed(keras.layers.Dense(28, activation = 'sigmoid'))
])

# Assemble the encoder

