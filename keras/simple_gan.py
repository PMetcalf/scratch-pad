'''
Simple GAN

Uses a generator and discriminator to develop hig-fidelity fake images of the MNIST fashion dataset.

'''

import keras

codings_size = 30

# Structure the generator
generator = keras.models.Sequential([
    keras.layers.Dense(100, activation = 'selu', input_shape = [codings_size]),
    keras.layers.Dense(150, activation = 'selu'),
    keras.layers.Dense(28 * 28, activation = 'sigmoid'),
    keras.layers.Reshape([28, 28])
])

# Structure the discriminator
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dense(150, activation = 'selu'),
    keras.layers.Dense(100, activaiton = 'selu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])