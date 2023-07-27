# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 15:28:43 2023

@author: saima
"""
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
%matplotlib inline
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from scipy.fftpack import dct
from scipy.fft import fft, dct
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
def compress_images(directory, quality):
    # 1. If there is a directory then change into it, else perform the next operations inside of the 
    # current working directory:
        count=0
    if directory:
        os.chdir(directory)

    # 2. Extract all of the .png and .jpeg files:
    files = os.listdir()

    # 3. Extract all of the images:
    images = [file for file in files if file.endswith(('jpg', 'png'))]

    # 4. Loop over every image:
    for image in images:
        if(count<= 500):
            print(image)

        # 5. Open every image:
            img = Image.open(image)

        # 5. Compress every image and save it with a new name: 
            img.save("Compressed_and_resized_with_function_"+image+quality, optimize=True, quality=quality)
            count=count+1

img_folder= "Path-to-fake-faces"
compress_images(img_folder, 10)
compress_images(img_folder, 30)
compress_images(img_folder, 50)
img_folder= "Path-to-real-faces"
compress_images(img_folder, 10)
compress_images(img_folder, 30)
compress_images(img_folder, 50)



def random_invert_img(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = (255-x)
  else:
    x
  return x
def random_invert(factor=0.5):
  return layers.Lambda(lambda x: random_invert_img(x, factor))

random_invert = random_invert()

from PIL import Image

def flipping(image):
    img_flip_ud_lr = cv2.flip(image, -1)
    return img_flip_ud_lr

def rotating(image): 
    img_rotate_90_counterclockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return img_rotate_90_counterclockwise
from PIL import Image, ImageEnhance
def change_contrast(img):
    dark_image = img - 50
    dark_image[dark_image < 0] = 0

    return dark_image


def adding_noise(img): 
    mean = 0.0   # some constant 
    std = 1.0    # some constant (standard deviation) 
    noisy_img = img + np.random.normal(mean, std, img.shape) 
    noisy_img_clipped = np.clip(noisy_img, 0, 255) 
    return  noisy_img_clipped


X_train=[]
Y_train=[]
freq_train=[]
Z_train=[]
epsilon = 1e-8
label=[]
datamask=[]
datafreq=[]
dataimg=[]
data=[]
i=0


folder1_path = "path-to-fake-image"
folder2_path = "path-to-fake-masks"

# Get a list of image filenames from each folder
folder1_images = [file for file in os.listdir(folder1_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
folder2_images = [file for file in os.listdir(folder2_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if both folders have the same number of images
if len(folder1_images) != len(folder2_images):
    print("Number of images in both folders must be the same.")
    exit()
count=0
# Read and process images from both folders simultaneously
for img_file1, img_file2 in zip(folder1_images, folder2_images):
    image_path1 = os.path.join(folder1_path, img_file1)
    image_path2 = os.path.join(folder2_path, img_file2)

    # Open images using PIL
    image = cv2.imread(image_path1)
    mask = cv2.imread(image_path2)
    if (count<=1000):
        random_invert_img(image, p=0.5)
        f = np.fft.fft2(t)
        m = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        fshift = np.fft.fftshift(f)
        #fshift += epsilon
        magnitude_spectrum = np.ceil(20*np.log(np.abs(fshift))).astype('float32') 
        t1 = flipping(t)
        f1 = np.fft.fft2(t1)
        fshift1 = np.fft.fftshift(f1)
        #fshift += epsilon
        magnitude_spectrum1 = np.ceil(20*np.log(np.abs(fshift1))).astype('float32') 
        m1 = flipping(m)
        t2 = rotating(t)
        f2 = np.fft.fft2(t2)
        fshift2 = np.fft.fftshift(f2)
        #fshift += epsilon
        magnitude_spectrum2 = np.ceil(20*np.log(np.abs(fshift2))).astype('float32') 
        m2 = rotating(m)
        t3=change_contrast(t)
        f3 = np.fft.fft2(t3)
        fshift3 = np.fft.fftshift(f3)
        #fshift += epsilon
        magnitude_spectrum3 = np.ceil(20*np.log(np.abs(fshift3))).astype('float32') 
        m3 = change_contrast(m)
        t4=adding_noise(t)
        f4 = np.fft.fft2(t4)
        fshift4 = np.fft.fftshift(f4)
        #fshift += epsilon
        magnitude_spectrum4 = np.ceil(20*np.log(np.abs(fshift4))).astype('float32') 
        m4 = change_contrast(m)
        xx = np.isfinite(magnitude_spectrum)
        image=np.array(t).astype('float32') 
        image1=np.array(t1).astype('float32') 
        image2=np.array(t2).astype('float32') 
        image3=np.array(t3).astype('float32') 
        image4=np.array(t4).astype('float32') 
        if (xx.all()==True):
            dataimg.append(image)
            dataimg.append(image1)
            dataimg.append(image2)
            dataimg.append(image3)
            dataimg.append(image4)
            datafreq.append(magnitude_spectrum)
            datafreq.append(magnitude_spectrum1)
            datafreq.append(magnitude_spectrum2)
            datafreq.append(magnitude_spectrum3)
            datafreq.append(magnitude_spectrum4)
            label.append(0)
            label.append(0)
            label.append(0)
            label.append(0)
            datamask.append(m)
            datamask.append(m1)
            datamask.append(m2)
            datamask.append(m3)
            datamask.append(m4)
    else:
        t=cv2.imread(os.path.join(img_folder, dir1)) 
        #image1=cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        t = random_invert(t)
        f = np.fft.fft2(t)
        t1 = random_invert(t)
        f = np.fft.fft2(t)
        fshift = np.fft.fftshift(f)
        #fshift += epsilon
        magnitude_spectrum = np.ceil(20*np.log(np.abs(fshift))).astype('float32') 
        #magnitude_spectrum=np.ceil(dct2(t))
        #magnitude_spectrum = random_invert(magnitude_spectrum)
        xx = np.isfinite(magnitude_spectrum)
        image=np.array(t).astype('float32') 
        if (xx.all()==True):
            dataimg.append(image)
            datafreq.append(magnitude_spectrum)
            label.append(0)
dataimg=np.array(dataimg)
datafreq=np.array(datafreq)
datamask=np.array(datamask)
            

W_train=datamask
X_train=dataimg
Y_train=datafreq
Z_train=label      

folder1_path = "path-to-real-image"
folder2_path = "path-to-real-masks"

# Get a list of image filenames from each folder
folder1_images = [file for file in os.listdir(folder1_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
folder2_images = [file for file in os.listdir(folder2_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if both folders have the same number of images
if len(folder1_images) != len(folder2_images):
    print("Number of images in both folders must be the same.")
    exit()
count=0
# Read and process images from both folders simultaneously
for img_file1, img_file2 in zip(folder1_images, folder2_images):
    image_path1 = os.path.join(folder1_path, img_file1)
    image_path2 = os.path.join(folder2_path, img_file2)

    # Open images using PIL
    image = cv2.imread(image_path1)
    mask = cv2.imread(image_path2)
    if (count<=1000):
        t = random_invert(t)
        f = np.fft.fft2(t)
        m = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        fshift = np.fft.fftshift(f)
        #fshift += epsilon
        magnitude_spectrum = np.ceil(20*np.log(np.abs(fshift))).astype('float32') 
        t1 = flipping(t)
        f1 = np.fft.fft2(t1)
        fshift1 = np.fft.fftshift(f1)
        #fshift += epsilon
        magnitude_spectrum1 = np.ceil(20*np.log(np.abs(fshift1))).astype('float32') 
        m1 = flipping(m)
        t2 = rotating(t)
        f2 = np.fft.fft2(t2)
        fshift2 = np.fft.fftshift(f2)
        #fshift += epsilon
        magnitude_spectrum2 = np.ceil(20*np.log(np.abs(fshift2))).astype('float32') 
        m2 = rotating(m)
        t3=change_contrast(t, 120)
        f3 = np.fft.fft2(t3)
        fshift3 = np.fft.fftshift(f3)
        #fshift += epsilon
        magnitude_spectrum3 = np.ceil(20*np.log(np.abs(fshift3))).astype('float32') 
        m3 = change_contrast(m,120)
        t4=adding_noise(t, 120)
        f4 = np.fft.fft2(t4)
        fshift4 = np.fft.fftshift(f4)
        #fshift += epsilon
        magnitude_spectrum4 = np.ceil(20*np.log(np.abs(fshift4))).astype('float32') 
        m4 = change_contrast(m,120)
        xx = np.isfinite(magnitude_spectrum)
        image=np.array(t).astype('float32') 
        image1=np.array(t1).astype('float32') 
        image2=np.array(t2).astype('float32') 
        image3=np.array(t3).astype('float32') 
        image4=np.array(t4).astype('float32') 
        if (xx.all()==True):
            dataimg.append(image)
            dataimg.append(image1)
            dataimg.append(image2)
            dataimg.append(image3)
            dataimg.append(image4)
            datafreq.append(magnitude_spectrum)
            datafreq.append(magnitude_spectrum1)
            datafreq.append(magnitude_spectrum2)
            datafreq.append(magnitude_spectrum3)
            datafreq.append(magnitude_spectrum4)
            label.append(1)
            label.append(1)
            label.append(1)
            label.append(1)
            datamask.append(m)
            datamask.append(m1)
            datamask.append(m2)
            datamask.append(m3)
            datamask.append(m4)
    else:
        t=cv2.imread(os.path.join(img_folder, dir1)) 
        #image1=cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        t = random_invert(t)
        f = np.fft.fft2(t)
        t1 = random_invert(t)
        f = np.fft.fft2(t)
        fshift = np.fft.fftshift(f)
        #fshift += epsilon
        magnitude_spectrum = np.ceil(20*np.log(np.abs(fshift))).astype('float32') 
        #magnitude_spectrum=np.ceil(dct2(t))
        #magnitude_spectrum = random_invert(magnitude_spectrum)
        xx = np.isfinite(magnitude_spectrum)
        image=np.array(t).astype('float32') 
        if (xx.all()==True):
            dataimg.append(image)
            datafreq.append(magnitude_spectrum)
            label.append(1)
dataimg=np.array(dataimg)
datafreq=np.array(datafreq)
datamask=np.array(datamask)
W_train=np.concatenate((W_train,mask), axis=0)
X_train=np.concatenate((X_train,data), axis=0)
Y_train=np.concatenate((Y_train,datafreq), axis=0)
Z_train=np.concatenate((Z_train,label), axis=0)

Z_train=to_categorical(Z_train, 2)
videodata="training-data.npz"
np.savez(videodata, W=W_train, X=X_train, Y=Y_train,Z=Z_train)


