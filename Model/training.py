# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 18:27:38 2023

@author: saima
"""

from Modeldefination import *
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
%matplotlib inline

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
#%matplotlib inline
n_filters = [32, 64, 128, 256, 512]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
shape=(224, 224, 3)
input1 = Input(shape)
input2 = Input(shape) 
   
(x,d1) = build_model(input1)
freq=freq_model(input2)
bili_reg = bili_regularizer_l2(strength=1)
bili_layer = bilinear_layer(num_outputs=2,
              channels_X=256,
              channels_Y=256,
              regularizer = bili_reg, 
              rank = 8,
              d = freq.shape[1] 
              )
out1 = bili_layer([freq, d1])
model = Model(inputs=[input1,input2], outputs=[x,out1])
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
def l2_loss(predictions, targets): 
    return tf.reduce_mean(tf.square(predictions - targets))
model.compile(optimizer=adam,
          loss={'x': 'l2_loss', 'out1': 'binary_crossentropy'},
          loss_weights={'x': 1, 'out1': 1 },
          metrics={'x': 'accuracy','out1': 'accuracy' })
model.summary()

videodata="training-data.npz"
loadeddata = np.load(videodata)
mask,images,freq_data,labels = loadeddata["W"],loadeddata["X"], loadeddata["Y"],loadeddata["Z"]

mask=mask.reshape([mask.shape[0],mask.shape[1],mask.shape[2],1])
import random 
def batch_generator(w,x, y,z, batch_size):
    sample_idx = random.randint(1, len(w))
    while True:
       W = np.zeros((batch_size, 224, 224, 3), dtype='float32')
       X = np.zeros((batch_size, 224, 224, 3), dtype='float32')
       y1 = np.zeros((batch_size, 224, 224,1), dtype='float32')
       y2 = np.zeros((batch_size, 1), dtype='float32')
       row=0
       # fill up the batch
       for row in range(batch_size):
           if (sample_idx > len(w)):
               sample_idx=0
           else:
               image = w[sample_idx] 
               freq_data=x[sample_idx] 
               mask = y[sample_idx]
               binary_value = z[sample_idx]
           # transform/preprocess image
    
               W[row, :, :, :] = image
               X[row, :, :, :] = freq_data
               y1[row, :, :,:] = mask
               y2[row, 0] = binary_value
               sample_idx += 1
               row+=1
       # Normalize inputs
    yield [W, X], {'output1': y1, 'output2': y2}
      
batch_size=16    
epoches=300  
early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=30, verbose=1)
model_checkpoint = ModelCheckpoint("E:/Uunet2dmedical.h5", monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
train_generator = batch_generator(images,freq_data,mask,labels, batch_size=batch_size)
val_generator = batch_generator(images,freq_data,mask,labels, batch_size=batch_size)
steps_per_epoch = len(images) // batch_size


model.fit_generator(generator=batch_generator(images, freq_data, mask, labels, batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epoches,  # Replace 10 with the desired number of epochs
                    validation_data=batch_generator(images, freq_data, mask, labels, 8),
                    callbacks=[early_stopping, model_checkpoint, reduce_lr],
                    verbose=1,
                    shuffle=True)