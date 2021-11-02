'''
Variational Autoencoder

Uses a Gaussian distribution between encoding and decoding layers to add a layer of filtering to inputs
and enable generation of simulated outputs.

Works with MNIST fashion dataset.
'''

import keras
from keras import activations
import tensorflow as tf

K = keras.backend

# Custom layer is used to sample codings
class Sampling(keras.layers.Layer):

    def call(self, inputs):

        mean, log_var = inputs

        # Calculate and return latent loss
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean

# Create the encoder, using Functional API as model is not entirely Sequential
codings_size = 10

inputs = keras.layers.Input(shape = [28, 28])
z = keras.layers.Flatten() (inputs)
z = keras.layers.Dense(150, activation = 'selu') (z)
z = keras.layers.Dense(100, activation = 'selu') (z)

# Note codings mean and log var both take inputs from last dense layer 
codings_mean = keras.layers.Dense(codings_size) (z)     # Mew ~ mean
codings_log_var = keras.layers.Dense(codings_size) (z)  # Gamma ~ Log of std squared

codings = Sampling() ([codings_mean, codings_log_var])

# Assemble the encoder
variational_encoder = keras.Model( inputs = [inputs], outputs = [codings_mean, codings_log_var, codings])

# Create the decoder, also using Functional API
decoder_inputs = keras.layers.Input(shape = [codings_size])
x = keras.layers.Dense(100, activation = 'selu') (decoder_inputs)
x = keras.layers.Dense(150, activation = 'selu') (x)
x = keras.layers.Dense(28 * 28, activation = 'sigmoid') (x)
outputs = keras.layers.Reshape([28, 28]) (x)

# Assemble the decoder
variational_decoder = keras.Model(inputs = [decoder_inputs], outputs = [outputs])