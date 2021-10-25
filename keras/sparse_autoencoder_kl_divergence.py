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