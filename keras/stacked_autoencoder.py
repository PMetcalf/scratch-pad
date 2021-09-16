'''
The following code builds a stacked autoencoder for Fashion MNIST (loaded and normalised) using the SELU activation function.
'''

from tensorflow import keras
from tensorflow.keras import optimizers

stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),   # Represents a 28 x 28 pixel picture, flattened into vector
    keras.layers.Dense(100, activation = 'selu'),   # LeCun normal initialisation could also be used here if the network was deeper
    keras.layers.Dense(30, activation = 'selu')
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation = 'selu', input_shape = [30]),
    keras.layers.Dense(28 * 28, activation = 'sigmoid'),
    keras.layers.Reshape([28, 28])      # Reshape the final vectors to match the encoder input
])

stacked_autoencoder = keras.models.Sequential([stacked_encoder, stacked_decoder])

optimizer = keras.optimizers.SGD(lr = 1.5)

# Binary cross-entropy loss is used as this is being treated as a multi-label classification problem
stacked_autoencoder.compile(loss = 'binary_crossentropy', optimizers = optimizer)

history = stacked_autoencoder.fit(X_train, X_train, epochs = 10, validation_data = [X_valid, X_valid])

'''

'''