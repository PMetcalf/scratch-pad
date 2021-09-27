'''
Convolutional Autoencoder for processing MNIST fashion dataset
'''

import keras

# Build the encoder
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape = [28, 28]),
    keras.layers.Conv2D(16, kernel_size = 3, padding = 'same', activation = 'selu'),
    keras.layers.MaxPool2D(pool_size = 2),
    ])

# Build the decoder

# Build the complete assembly