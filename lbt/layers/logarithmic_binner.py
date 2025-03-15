import tensorflow as tf
import numpy as np
import math
    
class LogarithmicWeightedBinner(tf.keras.layers.Layer):
    
    def __init__(self, key_refine=0):
        super().__init__()
        self.key_refine = key_refine
    
    def build(self, input_shape):
        assert len(input_shape) == 3 #(batch_size, input_len, n_feat)

        if self.key_refine >= 0:
            self.D = input_shape[2]
            self.N = input_shape[1]
            self.z = int(math.log(self.N, 2))
            self._refined_indices_init()
            
        self.built = True
        
    
    def _refined_indices_init(self):
        lb_refined_indices = np.append(np.array([1] * (2 ** self.key_refine - 1)), 
                                       np.repeat([2**i for i in range(self.z + 1)], repeats = 2 ** self.key_refine))
        
        lb_cumsum = np.cumsum(lb_refined_indices)
        lb_refined_indices = lb_refined_indices[:len(lb_cumsum[lb_cumsum <= self.N])]
        if self.N - sum(lb_refined_indices) > 0:
            lb_refined_indices = np.append(lb_refined_indices, self.N - sum(lb_refined_indices))
        lb_refined_indices = np.append(lb_refined_indices, 0)[::-1]
        
        assert sum(lb_refined_indices) == self.N
        assert len(lb_refined_indices) < self.N, "The value of key_refine is too large. "
        
        unique, _, counts = tf.unique_with_counts(lb_refined_indices)
        self.unique = tf.cast(unique, tf.int32)
        self.sum_counts = tf.cumsum(counts)
        self.lb_refined_indices = tf.convert_to_tensor(np.cumsum(lb_refined_indices), dtype=tf.int32)

        self.bin = []
        for i in range(1, len(self.unique)):
            self.bin.append(tf.keras.layers.Dense(1, use_bias = False))
        self.bias = self.add_weight(
            shape=(len(self.unique) - 1, self.D),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
    

    def call(self, inputs):
        if self.key_refine < 0:
            return inputs
        
        input_shape = tf.shape(inputs)
        outs = []
        for i in range(1, len(self.unique)):
            indices = self.lb_refined_indices
            out = inputs[:, indices[self.sum_counts[i - 1] - 1]:indices[self.sum_counts[i] - 1], :]
            out = tf.transpose(out, perm=[0, 2, 1])
            out = self.bin[i - 1](out)
            out = tf.transpose(out, perm=[0, 2, 1])
            outs.append(out)
        out = tf.concat(outs, axis=1)
        out += self.bias

        return tf.reshape(out, self.compute_output_shape(input_shape))
    
    def get_config(self):
        config = super().get_config()
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.lb_refined_indices) - 1, input_shape[-1])