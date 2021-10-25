'''
Sparse Autoencoder with KL (Kullback-Leibler) Divergence

Adds some regularisation to the coding layer's inputs to improve feature extraction per neuron, 
with tunable sparsity in the input layer, which seeks to balance the activation of neurons.

Setup to work with the MNIST clothing images dataset.
'''

import keras

K = keras.backend

kl_divergence = keras.losses.kullback_leibler_divergence