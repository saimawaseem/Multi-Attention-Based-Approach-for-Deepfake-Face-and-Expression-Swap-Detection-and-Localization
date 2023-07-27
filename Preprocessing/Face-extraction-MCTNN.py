# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:44:37 2023

@author: saima
"""

import pandas
import dlib
from mtcnn import MTCNN
import cv2
import os
import re
import json
from pylab import *
from PIL import Image, ImageChops, ImageEnhance
import scipy
import scipy.misc
from scipy import ndimage
import skimage.io
import skimage.filters
import sys
train_frame_folder = 'path-to-fake-video-folder'
training_percentage=0.3
ct=0
list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
for vid in list_of_train_data: 
    cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid)) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    training_frame_indexes = range(0, int(frame_count * training_percentage))  
    detector = MTCNN() 
    count = 0 
    ct = 0 
    while cap.isOpened():
     ret, frame = cap.read()
     if ret != True:
         break

     if ct in training_frame_indexes:
         face_rects = detector.detect_faces(frame)

         for i, d in enumerate(face_rects):
             x, y, width, height = face_rects[i]['box']
             x1, y1, x2, y2 = x - 10, y + 10, x - 10 + width + 20, y + 10 + height
             crop_img = frame[y1:y2, x1:x2]

             if crop_img.any():
                 output_path = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_{count}.png")
                 cv2.imwrite('path-to-save-images'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (224, 224)))
                 count+=1

     ct += 1
     print(ct)
cap.release()



###############################################################################################

train_frame_folder = 'path-to-real-video-folder'
training_percentage=0.7
ct=0
list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
for vid in list_of_train_data: 
    cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid)) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    training_frame_indexes = range(0, int(frame_count * training_percentage))  
    detector = MTCNN() 
    count = 0 
    ct = 0 
    while cap.isOpened():
     ret, frame = cap.read()
     if ret != True:
         break

     if ct in training_frame_indexes:
         face_rects = detector.detect_faces(frame)

         for i, d in enumerate(face_rects):
             x, y, width, height = face_rects[i]['box']
             x1, y1, x2, y2 = x - 10, y + 10, x - 10 + width + 20, y + 10 + height
             crop_img = frame[y1:y2, x1:x2]

             if crop_img.any():
                 output_path = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_{count}.png")
                 cv2.imwrite('path-to-save-images'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (224, 224)))
                 count+=1

     ct += 1
     print(ct)
cap.release()
