# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:39:49 2016

@author: aloha
"""


import softFunctions as sf
import allFunctions as af
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import cv2
from cv2 import *


# define params for selecting circle EDIT -> 'defineParamsForCircle(params)'
params_for_circle = cv2.SimpleBlobDetector_Params()
af.defineParamsForCircle(params_for_circle)
detector_for_circle = cv2.SimpleBlobDetector_create(params_for_circle)

# define range of blue color in HSV
# COLOR 1 | GREEN - moze i manje mnogo  - 40-70 npr
#lower_color = np.array([20,50,50])
#upper_color = np.array([100,255,255])

# COLOR 2 | ORANGE
lower_color = np.array([-100,50,50])
upper_color = np.array([20,255,255])

# COLOR 3 | BLUE
#lower_color = np.array([90,50,50])
#upper_color = np.array([160,255,255])


# helperq
x_old=0
y_old=0
crtam=0;

# blank image to draw on
blank_image = Image.new('RGBA', (500, 500), (255, 255, 255, 0))
blank_image_size=blank_image.size

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()
         
    #==================================================
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    #===================================================
    
    res=sf.erode(res)
    res, x, y = af.selectBlobFromG_v02(sf.invert(res), params_for_circle, detector_for_circle)  



    if(x==0 and y==0):              # blob nestao (x i y ==0)
        crtam=0
        x_old=0
        y_old=0
    else:                           # blob je tu(x ili y !=0), a pre toga nije bio tu (x_old  i y_old==0) -> spojice se tacka sama sa sobom
        crtam=1        
        if(x_old==0 and y_old==0):
            x_old=x
            y_old=y
                           
    if(crtam==1):
        af.connectDots(blank_image, x_old, y_old, x, y, blank_image_size[0])
        x_old=x
        y_old=y

        
    # open windows
    cv2.imshow('Camera',frame)
    #cv2.imshow('Mask',mask)
    cv2.imshow('Selected',res)
    cv2.imshow('CANVAS',np.asarray(blank_image.convert('RGB')))
    
    # Wait for exit (Q)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
#TODO - ima opcija slika.save() da sacuvamo to nacrtano