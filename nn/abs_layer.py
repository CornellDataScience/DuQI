import keras as k
from keras.engine.topology import Layer
import numpy as np

class AbsLayer(Layer):

    def __init__(self, **kwargs):
        super(AbsLayer, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     # Create a trainable weight variable for this layer.
    #     self.kernel = self.add_weight(name='kernel', 
    #                                   shape=(input_shape[1], self.output_dim),
    #                                   initializer='uniform',
    #                                   trainable=True)
    #     super(AbsLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return k.backend.abs(x)

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.output_dim)