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

# Build the decoder - Transpose convolutional layer (alternatively combine upsampling layers with convolutional layers)
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size = 3, strides = 2, padding = "valid", activation = "selu", input_shape = [3, 3, 61]),
    keras.layers.Conv2DTranspose(16, kernel_size = 3, strides = 2, padding = "same", activation = "sigmoid"),
    keras.layers.Reshape([28, 28])
])

# Build the complete assembly
conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])