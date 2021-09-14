'''
The following code builds a simple linear autoencoder to perform PCA on a 3D dataset, projecting it to 2D.
'''

from tensorflow import keras

encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape =[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape = [2])])

autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss = 'mse', optimizer = keras.optimizers.SGD(lr = 0.1))