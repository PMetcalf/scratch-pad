'''
The following code ties weights between encoder and decoder in a stacked autoencoder
'''

# Define a custom layer
class DenseTranspose(keras.layers.Layer):

# Acts like a regular dense layer, but uses another dense layer's weights

    def __init__(self, dense, activation = None, **kwargs):

        self.dense = dense

        self.activation = keras.activations.get(activation)

        super().__init__(**kwargs)

    def build(self, batch_input_shape):

        self.biases = self.add_weight(name = "bias",
                                        initializer = "zeros",
                                        shape = [self.dense.input_shape[-1]])

        super().build(batch_input_shape)

    def call(self, inputs):

        # Perform transposition of weights on the fly for efficiency using tf.matmul
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b = True)

        return self.activation(z + self.biases)