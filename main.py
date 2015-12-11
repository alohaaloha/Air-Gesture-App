# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:56:43 2015

Glavni program

@author: aloha
"""
import softFunctions as sf
import allFunctions as af

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame',gray)
    imgWithCircle=af.selectLight(frame)
    
    cv2.imshow('frame',imgWithCircle)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()