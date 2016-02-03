# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:39:49 2016

main prog

@author: aloha
"""

# ----------------------------------
import softFunctions as sf
import allFunctions as af
import nmFunctions as nmf

from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import cv2
from cv2 import *
import time
import nmFunctions as nm
# -----------------------------------


# define params for selecting circle EDIT -> 'defineParamsForCircle(params)'
params_for_circle = cv2.SimpleBlobDetector_Params()
af.defineParamsForCircle(params_for_circle)
detector_for_circle = cv2.SimpleBlobDetector_create(params_for_circle)

# define range of blue color in HSV
# COLOR 1 | GREEN - moze i manje mnogo  - 40-70 npr
lower_color1 = np.array([30,50,50])
upper_color1 = np.array([90,255,255])

# COLOR 2 | ORANGE
lower_color2 = np.array([-100,50,50])
upper_color2 = np.array([20,255,255])

# COLOR 3 | BLUE
#lower_color = np.array([90,50,50])
#upper_color = np.array([160,255,255])


# helper
x_old=0
y_old=0
crtam=0;


# blank image to draw on - we draw on this blank_image
blank_image = Image.new('RGBA', (500, 500), (255, 255, 255, 0))

# camera
cap = cv2.VideoCapture(0)

vreme = time.time()
canvasEmpty = True

while(1):

    # Take each frame
    _, frame = cap.read()
         
    # ==============================================================
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_color1, upper_color1)  # color 1
    mask2 = cv2.inRange(hsv, lower_color2, upper_color2) # color 2

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask) 
    res2 = cv2.bitwise_and(frame,frame, mask= mask2)
    # ===============================================================
    

    # ==================
    # trazi ZELENI BLOB - olovka
    # ==================
    res=sf.erode(res)
    res, x, y = af.selectBlobFromG_v02(sf.invert(res), params_for_circle, detector_for_circle)  

    if(x==0 and y==0):              # blob nije nadjen (x i y ==0), setuj old X, Y na 0
        crtam=0
        x_old=0
        y_old=0
    else:                           # blob je tu (x!=0 ili y!=0)
        crtam=1        
        if(x_old==0 and y_old==0):  # a ako pre toga nije bio tu (x_old==0  i y_old==0) => spojice se tacka sama sa sobom u 'connectDots()'
            x_old=x                 # inace ce se nova spojiti sa starom Xold i Yold 
            y_old=y
           

    # draw if blob is found (connectig old X, Y with new ones)
    if(crtam==1):
        af.connectDots(blank_image, x_old, y_old, x, y)
        x_old=x
        y_old=y
        canvasEmpty = False


    # =======================
    # trazi NARANDZASTI BLOB - pokazivac
    # na kopiji slike koja sluzi za crtanje(blank_image) ucrtavamo 'X' kao polozaj i tu kopiju prikazujemo
    # svaki frejm je nova kopija slike pa uvek prikazujemo poslednji polozaj 'X'
    # 'X' se crta u cosku ukoliko nije ukljucemo pozicioniranje, inace crta se gde se pronadjen
    # =======================
    x_cross=0
    y_cross=0
    res2=sf.erode(res2)
    ress, x_cross, y_cross = af.selectBlobFromG_v02(sf.invert(res2), params_for_circle, detector_for_circle)
    temp_img = af.drawCrossOnImg(blank_image.copy(), x_cross, y_cross)  
     

    # =============================
    # NM - slanje na prepoznavanje
    # ukoliko je vreme za to (2sek) - slika (blank_image) se salje na prepoznavanje
    # 
    # =============================





    # =============
    # open windows
    # =============
    cv2.imshow('Camera',frame)
    #cv2.imshow('Mask',mask)
    #cv2.imshow('Selected',res)
    #cv2.imshow('Selected',res2)
    cv2.imshow('Canvas',np.asarray(temp_img.convert('RGB')))
    
    # ================
    # na svake 4 sekunde ukoliko je pisano po canvasu
    # prosledjuje trenutnu sliku funkciji koja vraca prepoznat karakter sa slike
    # ================    
    if(time.time() > vreme + 4 and canvasEmpty == False):
        vreme = time.time()+4
        canvasEmpty = True
        nm.pogodi(np.asarray(blank_image.convert('RGB')))
        blank_image = Image.new('RGBA', (500, 500), (255, 255, 255, 0))    
    
    
    # Wait for exit (Q)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()