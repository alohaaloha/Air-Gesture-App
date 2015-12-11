# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:57:03 2015

@author: aloha
"""

import softFunctions as sf

import numpy as np
import cv2


def selectLight(img):
    gray=sf.image_gray(img)
    grayBlur=cv2.blur(gray, (10,10))
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grayBlur)
    cv2.circle(img, maxLoc, 5, (255, 0, 0), 2)
    return img, maxLoc
    
    
def selectBlob(img):
    return img