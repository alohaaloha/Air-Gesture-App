# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 04:07:42 2015
#INFO
# http://docs.opencv.org/master/d0/d7a/classcv_1_1SimpleBlobDetector.html#gsc.tab=0
# http://www.learnopencv.com/blob-detection-using-opencv-python-c/ <<--- ful yea ovo ono
@author: aloha
"""

import numpy as np
import cv2
from cv2 import *
import softFunctions as sf
import allFunctions as af
import matplotlib.pyplot as plt

#%%
img = sf.load_image("images/a.jpg")
sf.display_image(img)
#imwrite("filename.jpg",img) #save image

#%%
params_for_circle = cv2.SimpleBlobDetector_Params()
af.defineParamsForCircle(params_for_circle)
detector_for_circle = cv2.SimpleBlobDetector_create(params_for_circle)

#%%
#color = ([100, 180, 70], [140, 255, 130]) #GREEN
#color = ([120, 60, 60], [255, 120, 150]) #RED brightness LOW
#color = ([1, 130, 30], [120, 255, 200]) #GREEN low bri


#%%
frame=cv2.GaussianBlur(img, (5,5),0)
sl=af.selectColor(frame, color)
sl=sf.image_gray(sl)
sl=sf.invert(sl)
#sl=sf.image_bin_v02(sl)
#frameWithCircle, x, y = af.selectBlobFromG_v02(sl,params_for_circle, detector_for_circle)
frameWithCircle, x, y = af.selectBlobFromG(sl)
sf.display_image(frameWithCircle)
#sf.display_image(sl)

