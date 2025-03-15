import tensorflow as tf
import numpy as np
import math
import lbt
from lbt.layers.normalizer import *
from lbt.layers.logarithmic_binner import *

class Tesla(tf.keras.layers.Layer):
    def __init__(self, embed_dim = 8, num_head = 4, ff_dim = 64, key_refine=0, norm_min=0, norm_max=300):
        super().__init__()
        self.emb = tf.keras.layers.Dense(embed_dim)
        self.emb2 = tf.keras.layers.Dense(embed_dim)
        self.te = LogarithmicTransformerBlock(embed_dim, num_head, ff_dim)
        self.weighted_sum = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        data, feature = inputs[..., :1], inputs[..., 1:]
        iout = self.emb2(tf.transpose(data, perm=[0, 2, 1])) #B, 1, D
        out = self.emb(data) #B, S, D
        out = tf.concat([out + iout], axis=1) #B, (S+1), D
        out = self.te(out) #B, S, D
        out = out[:, -1]
        out = self.weighted_sum(out)
        return out
    
    def get_config(self):
        config = super().get_config()
        return config

class LogarithmicTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim = 8, num_heads = 4, ff_dim = 128, key_refine=0):
        super().__init__()
        self.lb = LogarithmicWeightedBinner(0)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.layers.Dense(1)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs):
        out1 = self.lb(inputs)
        attn_output, self.attn_scores = self.att(out1, out1, return_attention_scores=True)
        out2 = self.layernorm1(out1 + attn_output)
        out2 = tf.transpose(out2, perm=[0, 2, 1])
        return tf.transpose(self.ffn(out2), perm=[0, 2, 1])
    
    def get_config(self):
        config = super().get_config()
        return config