'''
The following code builds a stacked autoencoder for Fashion MNIST (loaded and normalised) using the SELU activation function.
'''

from tensorflow import keras

stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dense(100, activation = 'selu'),
    keras.layers.Dense(30, activation = 'selu')
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation = 'selu', input_shape = [30]),
    keras.layers.Dense(28 * 28, activation = 'sigmoid'),
    keras.layers.Reshape([28, 28])      # Reshape the dense layer output to match the encoder input
])