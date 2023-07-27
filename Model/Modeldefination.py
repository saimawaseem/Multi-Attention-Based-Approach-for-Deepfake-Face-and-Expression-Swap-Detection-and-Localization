# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:27:54 2023

@author: saima
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:04:52 2023

@author: saima
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf

import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, concatenate, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers, callbacks
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras import layers
#%matplotlib inline
n_filters = [32, 64, 128, 256, 512]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
def ca_stem_block(inputs, filters, strides=1):
    """
    Residual block for the first layer of Deep Residual U-Net.
    See: https://arxiv.org/pdf/1711.10684.pdf
    Code from: https://github.com/dmolony3/ResUNet
    """
    # Conv
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)

    # CA
    x = ca_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def feature_fusion(high, low):
    """
    Low- and high-level feature fusion, taking advantage of multi-level contextual information.
    Args:
        high: high-level semantic information in the contracting path.
        low: low-level feature map in the symmetric expanding path.
    See: https://arxiv.org/pdf/1804.03999.pdf
    """
    filters = low.shape[-1]

    x1 = UpSampling2D(size=(2, 2))(high)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(filters, (3, 3), padding="same")(x1)

    x2 = BatchNormalization()(low)
    x2 = Activation("relu")(x2)
    x2 = Conv2D(filters, (3, 3), padding="same")(x2)

    x = Add()([x1, x2])

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)

    outputs = Multiply()([x, low])
    return outputs


def ca_block(inputs, ratio=16):
    """
    Channel Attention Module exploiting the inter-channel relationship of features.
    """
    shape = inputs.shape
    filters = shape[-1]

    # avg_pool = Lambda(lambda x: K.mean(x, axis=[1, 2], keepdims=True))(inputs)
    # max_pool = Lambda(lambda x: K.max(x, axis=[1, 2], keepdims=True))(inputs)
    # avg_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    # max_pool = MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
    avg_pool = K.mean(inputs, axis=[1, 2], keepdims=True)
    max_pool = K.max(inputs, axis=[1, 2], keepdims=True)
    
    x1 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(avg_pool)
    x1 = Dense(filters, activation=None, kernel_initializer='he_normal', use_bias=False)(x1)
    
    x2 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(max_pool)
    x2 = Dense(filters, activation=None, kernel_initializer='he_normal', use_bias=False)(x2)
    

    x = Add()([x1, x2])
    x = Activation("sigmoid")(x)
    
    outputs = Multiply()([inputs, x])
    return outputs

def sa_block(inputs):
    """
    Spatial Attention Module utilizing the inter-spatial relationship of features.
    """
    kernel_size = 7

    # avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    # max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    avg_pool = K.mean(inputs, axis=-1, keepdims=True)
    max_pool = K.max(inputs, axis=-1, keepdims=True)

    x = Concatenate()([avg_pool, max_pool])

    x = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)

    outputs = Multiply()([inputs, x])
    return outputs


def cbam_block(inputs):
    """
    CBAM: Convolutional Block Attention Module, which combines Channel Attention Module and Spatial Attention Module,
    focusing on `what` and `where` respectively. The sequential channel-spatial order proves to perform best.
    See: https://arxiv.org/pdf/1807.06521.pdf
    """
    x = ca_block(inputs)
    x = sa_block(x)
    return x


def res_block(inputs, filters, strides=1):
    """
    Residual block with full pre-activation (BN-ReLU-weight-BN-ReLU-weight).
    See: https://arxiv.org/pdf/1512.03385.pdf & https://arxiv.org/pdf/1603.05027v3.pdf
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def ca_resblock(inputs, filters, strides=1):
    """
    Residual block with Channel Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # CA
    x = ca_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def sa_resblock(inputs, filters, strides=1):
    """
    Residual block with Spatial Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # SA
    x = sa_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs

def cbam_resblock(inputs, filters, strides=1):
    """
    Residual block with Convolutional Block Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # CBAM
    x = cbam_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs

def encode_decode(encode,decode,n_filters, strides=1):
       e1 = Concatenate()([encode, encode])
       d1= Concatenate()([decode, decode])
       encode1=cbam_resblock(e1,n_filters)
       decode1=cbam_resblock(d1,n_filters)
       outputs = Add()([encode1, decode1])
       shape = outputs.shape
       filters = shape[-1]
       outputs = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(outputs)
       return outputs
       

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply

def build_model(inputs, num_classes=1):
    """
    Build a model with fixed input shape [N, H, W, C].
    """
    n_filters = [32, 64, 128, 256, 512]

    #inputs = Input(shape)

    # Encoder
    c0 = ca_stem_block(inputs, n_filters[0])

    c1 = res_block(c0, n_filters[1], strides=1)
    c1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)

    c2 =  res_block(c1, n_filters[2], strides=1)
    c2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)

    c3 =  res_block(c2, n_filters[3], strides=1)
    c3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c3)

    # Bridge
    b1 = res_block(c3, n_filters[4])
    b1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(b1)
    # Decoder
    # Nearest-neighbor UpSampling followed by Conv2D & ReLU to dampen checkerboard artifacts.
    # See: https://distill.pub/2016/deconv-checkerboard/

    d1 = UpSampling2D(size=(2, 2))(b1)
    d1 = Conv2D(n_filters[3], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d1)
    d1_2=encode_decode(c3,d1,n_filters[3])
    #d1 = feature_fusion(c3, d1)
    d100 = Multiply()([c3, d1_2])
    d10 = Concatenate()([d100, d1])
    shape = d100.shape
    filters = shape[-1]
    d10= BatchNormalization()(d10)
    d10 = Activation('relu')(d10)
    d10 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(d10)

    d2 = UpSampling2D(size=(2, 2))(d10)
    d2 = Conv2D(n_filters[2], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d2)
    d2_2=encode_decode(c2,d2,n_filters[2])
    d200 = Multiply()([c2, d2_2])
    d20 = Concatenate()([d200, d2])
    shape = d200.shape
    filters = shape[-1]
    d20= BatchNormalization()(d20)
    d20 = Activation('relu')(d20)
    d20 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(d20)
    #d2 = feature_fusion(c2, d2)


    d3 = UpSampling2D(size=(2, 2))(d20)
    d3 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d3)
    #d3 = feature_fusion(c1, d3)
    d3_2=encode_decode(c1,d3,n_filters[1])
    d300 = Multiply()([c1, d3_2])
    d30 = Concatenate()([d300, d3])
    shape = d300.shape
    filters = shape[-1]
    d30= BatchNormalization()(d30)
    d30 = Activation('relu')(d30)
    d30 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(d30)

    d4 = UpSampling2D(size=(2, 2))(d30)
    d4 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d4)
    #d3 = feature_fusion(c1, d3)
    d4_2=encode_decode(c0,d4,n_filters[0])
    d400 = Multiply()([c0, d4_2])
    d40 = Concatenate()([d400, d4])
    shape = d400.shape
    filters = shape[-1]
    d40= BatchNormalization()(d40)
    d40 = Activation('relu')(d40)
    d40 = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(d40)

    # Output
    outputs = ca_resblock(d40, n_filters[0])
    outputs = Conv2D(num_classes, (1, 1), padding="same")(outputs)
    outputs = Activation("sigmoid")(outputs)

    # Model
    #model = Model(inputs, outputs)
    return outputs,d10
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply

def attention_block(input_tensor):
    """
    Simple attention block with Global Average Pooling and Dense block.
    """
    # Global Average Pooling
    attention = GlobalAveragePooling2D()(input_tensor)

    # Dense block
    attention = Dense(units=256, activation='relu')(attention)
    attention = Dense(units=1, activation='sigmoid')(attention)

    # Apply attention
    attention = Multiply()([input_tensor, attention])

    return attention
def freq_model(inputs):
    """
    Build a model with fixed input shape [N, H, W, C].
    """
    n_filters = [32, 64, 128, 256, 512]


    # Encoder
    m0 = ca_stem_block(inputs, n_filters[0])

    m1 = res_block(m0, n_filters[1], strides=1)
    m1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m1)

    m2 =  res_block(m1, n_filters[2], strides=1)
    m2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m2)

    m3 =  res_block(m2, n_filters[3], strides=1)
    m3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(m3)
    m4=attention_block(m3)
    return m4
class bili_regularizer_l2(regularizers.Regularizer):
    '''
        Standard L2 regularization applied to the weight matrix for the bilinear layer.
    '''
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, weights):
        w0 = weights[0]
        w1 = weights[1]
        # d = 3 # w0.shape[2]
        #z = tf.norm(w0, ord=2) + tf.norm(w1, ord=2) 
        T1 = tf.matmul(tf.transpose(w0, perm=[0,2,1]), w0)
        T2 = tf.matmul(tf.transpose(w1, perm=[0,2,1]), w1) 
        z = tf.linalg.trace(tf.matmul(T1, T2))
        # print("z = ", tf.reduce_sum(z))
        return self.strength * tf.reduce_sum(z) 
class bilinear_layer(Layer):
    def __init__(self, num_outputs, channels_X, channels_Y, regularizer, d, rank, seed=1):
        super(bilinear_layer, self).__init__()
        self.num_outputs = num_outputs
        self.channels_X = channels_X # the number of features in the first NN (m in the blog post)
        self.channels_Y = channels_Y # the number of features in the second NN (n in the blog post)
        self.d = d # the dimensionality of the feature maps
        self.rank = rank # the rank of the low-rank matrices
        self.kernel_regularizer = regularizer 

    def build(self, input_shape):
        
        tf.random.set_seed(1)
        self.w = self.add_weight(shape=(2, self.num_outputs, self.channels_X, self.rank),
                                    initializer="random_normal",
                                    trainable=True,
                                    regularizer=self.kernel_regularizer)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.num_outputs,), dtype="float32"), trainable=True
        )
        
    def call(self, inputs):
        X, Y = inputs[0], inputs[1]       
        X = tf.reshape(X, (-1, 1, self.channels_X, self.d*self.d)) 
        Y = tf.reshape(Y,  (-1, 1, self.channels_Y, self.d*self.d)) 
        T1 = tf.matmul(tf.transpose(X, perm=[0,1,3,2]), self.w[0])
        T2 = tf.matmul(tf.transpose(self.w[1], perm=[0,2,1]), Y) 
        
        # The matrix trace takes only the diagonal entries. 
        # The expression below computes sum_{i=1}^d u_i^TW_k v_i from the blog post
        z = tf.linalg.trace(tf.matmul(T1, T2))/(self.d*self.d) + self.b
        softmax = tf.keras.layers.Softmax()
        z = softmax(z)
        return z 
    
    


    

    
    
    
    
