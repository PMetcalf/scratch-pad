'''
Sparse Autoencoder

Adds some regularisation to the coding layer's inputs to improve feature extraction per neuron.

Setup to work with the MNIST clothing images dataset.
'''

import keras

# Create the regularised encoder
sparse_l1_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dense(100, activation = 'selu'),
    keras.layers.Dense(300, activation = 'sigmoid'),    # This layer is regularised to improve feature extraction
    keras.layers.ActivityRegularization(l1 = 1e-3)      # This regularisation improves feature extraction
])

# Create the decoder
sparse_l1_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation = 'selu', input_shape = [300]),  # Work with the regularised layer
    keras.layers.Dense(28 * 28, activation = 'sigmoid'),
    keras.layers.Reshape([28, 28])
])

# Build the autoencoder
sparse_l1_ae = keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])