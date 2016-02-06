# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:56:43 2015

main02

@author: aloha
"""

#%% IMPORT LIBS

import softFunctions as sf
import allFunctions as af
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import cv2


#%% SET

# define params
params_for_circle = cv2.SimpleBlobDetector_Params()
params_for_triangle = cv2.SimpleBlobDetector_Params()
af.defineParamsForCircle(params_for_circle)
af.defineParamsForTriangle(params_for_triangle)
# Create a detector with the parameters
detector_for_circle = cv2.SimpleBlobDetector_create(params_for_circle)
detector_for_triangle = cv2.SimpleBlobDetector_create(params_for_triangle)



#TODO - velicina kamere
blank_image = Image.new('RGBA', (500, 500), (255, 255, 255, 0))
#blank_image = Image.open("BLANK.png")
blank_image_size=blank_image.size


cap = cv2.VideoCapture(0)
#help stuff
x_old=0
y_old=0
crtam=0;

while(True):
    
    # 1. Capture frame-by-frame
    ret, frame = cap.read()

    # 2. Our operations on the frame come here    
    frameWithCircle, x, y = af.selectBlob_v2(frame, params_for_circle, detector_for_circle)
    state=1 # u frejmu je krug
    if(x==0 and y==0):
        frameWithCircle, x, y = af.selectBlob_v2(frame, params_for_triangle, detector_for_triangle)
        state=2 # u frejmu je trougao
        if(x==0 and y==0):
            state=3 # u frejmu je nista
    
    
    #naci nas kad nem blage sta sam ovde uradio
    if(x==0 and y==0):
        crtam=0
        x_old=0
        y_old=0
    else:
        crtam=1        
        if(x_old==0 and y_old==0):
            x_old=x
            y_old=y
                           
    if(crtam==1 and state==2):
        af.connectDots(blank_image, x_old, y_old, x, y, blank_image_size[0])
        x_old=x
        y_old=y

    
    # 3. Display the resulting frame
    cv2.imshow('CANVAS',np.asarray(blank_image.convert('RGB')))
    cv2.imshow('CAMERA', frameWithCircle)
    #print blank_image_size[0]
    #print frame.size
         
    # 4. Wait for exit   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#TODO - ima opcija slika.save() da sacuvamo to nacrtano