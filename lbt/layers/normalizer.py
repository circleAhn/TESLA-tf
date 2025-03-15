import tensorflow as tf
import numpy as np
import math

class Normalizer(tf.keras.layers.Layer):
    def __init__(self, shift = 0., scale = 300.):
        super().__init__()
        self.shift = shift
        self.scale = scale
        
    def call(self, inputs):
        inputs += self.shift
        return inputs / self.scale
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale
        })
        return config
    
class Denormalizer(tf.keras.layers.Layer):
    def __init__(self, shift = 0., scale = 300.):
        super().__init__()
        self.shift = shift
        self.scale = scale
        
    def call(self, inputs):
        return inputs * self.scale - self.shift
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale
        })
        return config