# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:21:45 2015

@author: aloha
"""
 #%% IMPORT ALL LIBRARIES
import softFunctions as sf

import numpy as np
import cv2


#%% import test img
image = sf.load_image('roiTest.jpg')
orig=image.copy()
blur=cv2.blur(image,(5,5)) 
gray=sf.image_gray(image)
grayBlur=cv2.blur(gray, (5,5))

sf.display_image(grayBlur)

#%% METOD 0 - probati sa select_roi ovo ono
img = sf.image_bin_adaptive(sf.image_gray(image))
img =sf.invert(img)
sf.display_image(img)


#%% METOD 1
# the area of the image with the largest intensity value
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayBlur)
img=orig.copy()
cv2.circle(img, maxLoc, 5, (255, 0, 0), 2)

sf.display_image(img)
print maxLoc

#%% METOD 2