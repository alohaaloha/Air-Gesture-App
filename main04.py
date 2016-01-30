# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:56:43 2015

main02
q
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
from cv2 import *


#%% SET STUFF FOR SELECTING
#### define params for CIRCLE and TRIANGLE
params_for_circle = cv2.SimpleBlobDetector_Params()
af.defineParamsForCircle(params_for_circle)
#### Create a detector with the parameters for CIRCLE
detector_for_circle = cv2.SimpleBlobDetector_create(params_for_circle)
#### colors
hue = 18

#%% SET WINDOW
#TODO - get velicina kamere
blank_image = Image.new('RGBA', (500, 500), (255, 255, 255, 0))
#blank_image = Image.open("BLANK.png")
blank_image_size=blank_image.size


#### help stuff
x_old=0
y_old=0
crtam=0;

#### camera
cap = cv2.VideoCapture(0)

while(True):
    
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    

    bin_img=af.selectColorHSV(80, frame)

    bin_img=sf.erode(bin_img)
    bin_img=sf.erode(bin_img)

    for i in range(0,10):
        bin_img=sf.dilate(bin_img)


    #frameWithCircle, x, y = af.selectBlobFromG(sf.invert(bin_img))  
    frameWithCircle, x, y = af.selectBlobFromG_v02(sf.invert(bin_img),  params_for_circle, detector_for_circle)

    
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