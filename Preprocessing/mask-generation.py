# -*- coding: utf-8 -*-
                                                                                                                                                                                                                                     """
Created on Sat Mar 12 15:40:09 2022

@author: CVVIP
"""

import argparse
import cv2
import numpy as np
import os
import sys
imageSize=224
input_fake="path-to-faceswap-directory-of-videos"
mask="path-to-faceswap-directory-of-mask-videos"
output_fake="output"
scale=1.3


def to_bw(mask, thresh_binary=10, thresh_otsu=255):
    im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, thresh_binary, thresh_otsu, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return im_bw

def get_bbox(mask, thresh_binary=127, thresh_otsu=255):
    im_bw = to_bw(mask, thresh_binary, thresh_otsu)

    # im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    locations = np.array([], dtype=np.int).reshape(0, 5)

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
        else:
            cX = 0
        if M["m00"] > 0:
            cY = int(M["m01"] / M["m00"])
        else:
            cY = 0

        # calculate the rectangle bounding box
        x,y,w,h = cv2.boundingRect(c)
        locations = np.concatenate((locations, np.array([[cX, cY, w, h, w + h]])), axis=0)

    max_idex = locations[:,4].argmax()
    bbox = locations[max_idex, 0:4].reshape(4)
    return bbox

def extract_face(image, bbox, scale = 1.0):
    h, w, d = image.shape
    radius = int(bbox[3] * scale / 2)

    y_1 = bbox[1] - radius
    y_2 = bbox[1] + radius
    x_1 = bbox[0] - radius
    x_2 = bbox[0] + radius

    if x_1 < 0:
        x_1 = 0
    if y_1 < 0:
        y_1 = 0
    if x_2 > w:
        x_2 = w
    if y_2 > h:
        y_2 = h

    crop_img = image[y_1:y_2, x_1:x_2]

    if crop_img is not None:
        crop_img = cv2.resize(crop_img, (imageSize, imageSize))

    return crop_img

####################masks for face-swap videos###########
training_percentage=0.3

for f in os.listdir(input_fake):
     if os.path.isfile(os.path.join(input_fake, f)):
         if f.lower().endswith(('mp4')):
             print(f)
             filename = os.path.splitext(f)[0]
             vidcap_fake = cv2.VideoCapture(os.path.join(input_fake, f))
             frame_count = int(vidcap_fake.get(cv2.CAP_PROP_FRAME_COUNT)) 
             training_frame_indexes = range(0, int(frame_count * training_percentage)) 
             success_fake, image_fake = vidcap_fake.read()

             image_mask = cv2.imread('0000.jpg')
             count = 0

             while (success_fake):

                 bbox = get_bbox(image_mask)
                 altered_cropped = extract_face(image_fake, bbox, scale)

                 mask_cropped = to_bw(extract_face(image_mask, bbox, scale))
                 mask_cropped = cv2.bitwise_not(mask_cropped) 
                 mask_cropped = np.stack((mask_cropped,mask_cropped, mask_cropped), axis=2) 
                 if (altered_cropped is not None) and (mask_cropped is not None):
                     cv2.imwrite(os.path.join(output_fake_image, filename + "_%d.jpg" % count), altered_cropped)
                     
                     count += 1

                 if count >= (list(training_frame_indexes)[0] + 1):
                     break

                 success_fake, image_fake = vidcap_fake.read()


####################masks for fake expression videos###########
f_vid_mask = "path-to-FF++-expression-swap-maskvideos"
f_vid_altered = "path-to-FF++-expression-swap-videos"
count=0
f_img_altered = "output-path-to-save-mask-images"
training_percentage=0.3
cout=0
for f in os.listdir(f_vid_mask):
    if os.path.isfile(os.path.join(f_vid_mask, f)):
        if f.lower().endswith(('mp4')):
            print(f)
            filename = os.path.splitext(f)[0]

            vidcap_mask = cv2.VideoCapture(os.path.join(f_vid_mask, f))
            success_mask, image_mask = vidcap_mask.read()

            vidcap_altered = cv2.VideoCapture(os.path.join(f_vid_altered, f))
            success_altered, image_altered = vidcap_altered.read()
            frame_count = int(vidcap_altered.get(cv2.CAP_PROP_FRAME_COUNT)) 
            training_frame_indexes = range(0, int(frame_count * training_percentage)) 
            count = 0

            while (success_altered):
                bbox = get_bbox(image_mask)

                if bbox is None:
                    count += 1
                    continue
                mask_cropped = to_bw(extract_face(image_mask, bbox, scale))
                mask_cropped = cv2.bitwise_not(mask_cropped)
                mask_cropped = np.stack((mask_cropped,mask_cropped, mask_cropped), axis=2)
                

                if  (altered_cropped is not None):
                    cv2.imwrite(os.path.join(f_img_altered, filename + "_%d.jpg" % count), mask_cropped)
                    count=count+1
                if count >= (list(training_frame_indexes)[0] + 1): 
                    break

                success_mask, image_mask = vidcap_mask.read()
                success_altered, image_altered = vidcap_altered.read()




####################masks for original videos###########
s=os.listdir(f_vid_mask)
i=0
count=0
input_real="inputpath-to-real-videos"
f_img_original="outputpath-to-saving-masks"
training_percentage=0.7
blank_img = np.zeros((224,224,3), np.uint8)
for f in os.listdir(input_real):
    if os.path.isfile(os.path.join(input_real, f)):
        if f.lower().endswith(('mp4')):
            print(f)
            filename = os.path.splitext(f)[0]
            vidcap_original = cv2.VideoCapture(os.path.join(input_real, f))
            frame_count = int(vidcap_original.get(cv2.CAP_PROP_FRAME_COUNT)) 
            training_frame_indexes = range(0, int(frame_count * training_percentage)) 
            success_original, image_original = vidcap_original.read()
            count = 0

            while (success_original):
                    original_cropped = blank_img
                    mask_cropped = cv2.bitwise_not(original_cropped)
                    if (image_original is not None): 
                        cv2.imwrite(os.path.join(f_img_original, filename + "_%d.jpg" % count), mask_cropped) 
                        count += 1
                    if count >= (list(training_frame_indexes)[0] + 1): 
                        break
                    success_original, image_original = vidcap_original.read()


