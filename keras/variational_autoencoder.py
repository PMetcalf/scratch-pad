'''
Variational Autoencoder

Uses a Gaussian distribution between encoding and decoding layers to add a layer of filtering to inputs
and enable generation of simulated outputs.

Works with MNIST fashion dataset.
'''

import keras
import tensorflow as tf

K = keras.backend

# Custom layer is used to sample codings
class Sampling(keras.layers.Layer):

    def call(self, inputs):

        mean, log_var = inputs

        # Calculate and return latent loss
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean