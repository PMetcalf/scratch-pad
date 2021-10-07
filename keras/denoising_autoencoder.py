'''
Denoising Autoencoder 

Regular stacked autoencoder with dropout applied to the encoder's inputs.

Training focuses on recovering original, noise-free inputs.
'''

import keras

# Dropout encoder
dropout_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dropout(0.5),                      # Dropout layers adds noise to incoming signal
    keras.layers.Dense(100, activation = 'selu'),
    keras.layers.Dense(30, activation = 'selu')
])

# Dropout decoder
dropout_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation = 'selu', input_shape = [30]),
    keras.layers.Dense(28 * 28, activation = 'sigmoid'),
    keras.layers.Reshape([28, 28])
    ])

# Full autoencoder
dropout_ae = keras.models.Sequential([dropout_encoder, dropout_decoder])