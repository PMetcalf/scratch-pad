'''
Sparse Autoencoder with KL (Kullback-Leibler) Divergence

Adds some regularisation to the coding layer's inputs to improve feature extraction per neuron, 
with tunable sparsity in the input layer, which seeks to balance the activation of neurons.

Setup to work with the MNIST clothing images dataset.
'''

import keras

K = keras.backend

kl_divergence = keras.losses.kullback_leibler_divergence

# Create custom regulariser to apply KL divergence regularisation
class KLDivergenceRegulariser(keras.regularizers.Regularizer):

    def __init__(self, weight, target = 0.1):
        
        self.weight = weight
        self.target = target

    def __call__(self, inputs):

        mean_activities = K.mean(inputs, axis = 0)

        return self.weight * (
            kl_divergence(self.target, mean_activities) + kl_divergence(1. - self.target, 1. - mean_activities)
        )

# Build the regulariser
kld_regulariser = KLDivergenceRegulariser(weight = 0.05, target = 0.1)

# Build the encoder, with regulariser applied to final dense layer
sparse_kl_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dense(100, activation = 'selu'),
    keras.layers.Dense(300, activation = 'sigmoid', activity_regularizer = kld_regulariser)
])

# Build the decoder
sparse_kl_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation = 'selu', input_shape = [300]),
    keras.layers.Dense(28 * 28, activation = 'sigmoid'),
    keras.layers.Reshape([28, 28])
])

# Build the autoencoder
sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])