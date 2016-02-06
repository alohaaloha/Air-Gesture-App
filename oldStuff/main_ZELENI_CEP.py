# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:56:43 2015

Glavni program

@author: aloha
"""
import softFunctions as sf
import allFunctions as af
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import nmFunctions as nm


#TODO - velicina kamere
blank_image = Image.new('RGBA', (500, 500), (255, 255, 255, 0))
brush = Image.open('brush.png')
#blank_image = Image.open("BLANK.png")
blank_image_size=blank_image.size

cap = cv2.VideoCapture(0)

#help stuff
x_old=0
y_old=0
crtam=0;

vreme = time.time()
canvasEmpty = True

while(True):
    
    # 1. Capture frame-by-frame
    ret, frame = cap.read()

    # 2. Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame,1)
    frameWithCircle, x, y ,state= af.selectBlob2(frame)
    
    
    if(state == 1):
        if(x_old==0 and y_old==0):
            x_old=x
            y_old=y
        af.connectDots(blank_image, x_old, y_old, x, y)
        x_old=x
        y_old=y
        canvasEmpty = False
    else:
        x_old=0
        y_old=0


    #temp.paste(brush,(x,y),brush)
    temp = af.drawCrossOnImg(blank_image.copy(),x,y)
    #temp = af.drawBrush(blank_image.copy(),brush.copy(),x,y)
    # 3. Display the resulting frame
    #cv2.imshow('prikaz frame-a',gray)
    cv2.imshow('CANVAS', np.asarray(temp.convert('RGB'))) #np.asarray(blank_image.convert('RGB'))
    cv2.imshow('CAMERA', frameWithCircle)
    #cv2.imshow('CAMERA', frame)
    #print blank_image_size[0]
    #print frame.size
    if(time.time() > vreme + 5 and canvasEmpty == False):
        vreme = time.time()+5
        canvasEmpty = True
        nm.pogodi(np.asarray(blank_image.convert('RGB')))
        blank_image = Image.new('RGBA', (500, 500), (255, 255, 255, 0))
      
    # 4. Wait for exit   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#TODO - ima opcija slika.save() da sacuvamo to nacrtano