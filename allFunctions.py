# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:57:03 2015

Fajl sadrzi funkcije koriscenje za slektovanje roi, crtanje na slici, podesavanje parametara selekcije

@author: aloha
"""

import softFunctions as sf
import numpy as np
import cv2
from PIL import Image, ImageDraw
import numpy as np

    
# SELECTION 
    
def selectBlob(img):
    
    gray=sf.image_gray(img)
     
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()
     
    # Detect blobs.
    #print dir(keypoints)
    keypoints = detector.detect(gray)
    if not keypoints:
        x=0
        y=0
    else:
        x=keypoints[0].pt[0]
        y=keypoints[0].pt[1]   
    
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    # Show keypoints
    #sf.display_image(im_with_keypoints)      
    return im_with_keypoints, x, y
    
    
def selectBlobFromG(img):
    
    gray=img
     
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()
     
    # Detect blobs.
    #print dir(keypoints)
    keypoints = detector.detect(gray)
    if not keypoints:
        x=0
        y=0
    else:
        x=keypoints[0].pt[0]
        y=keypoints[0].pt[1]   
    
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    # Show keypoints
    #sf.display_image(im_with_keypoints)      
    return im_with_keypoints, x, y    
    
    

def selectBlobFromG_v02(img, params, detector):
    
    gray=img
     
     
    # Detect blobs.
    #print dir(keypoints)
    keypoints = detector.detect(gray)
    if not keypoints:
        x=0
        y=0
    else:
        x=keypoints[0].pt[0]
        y=keypoints[0].pt[1]   
    
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    # Show keypoints
    #sf.display_image(im_with_keypoints)      
    return im_with_keypoints, x, y 
    


def selectBlob_v2(image, params, detector):
    gray=sf.image_gray(image)
    #gray=sf.invert(gray)
    keypoints = detector.detect(gray)
    
    if not keypoints:
        x=0
        y=0
    else:
        x=keypoints[0].pt[0]
        y=keypoints[0].pt[1]      
    
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints, x, y
    
        

# DRAWING
    
    
def drawDotOnImg(blank_image, x, y, width):
    draw = ImageDraw.Draw(blank_image)
    draw.ellipse((width-x, y, width-x+10, y+10), fill = 'red', outline ='red')
        #draw.line((100, 200, 150, 300), fill=500)
    
    
def connectDots(blank_image, x1, y1, x2, y2):
    blank_image_size=blank_image.size
    width=blank_image_size[0]
    draw = ImageDraw.Draw(blank_image)
    draw.line((width-x1, y1, width-x2, y2), fill=500, width=10)
    
def drawCrossOnImg(blank_image, x, y):
    draw = ImageDraw.Draw(blank_image)
    #draw.ellipse((x, y, x+10, y+10), fill = 'red', outline ='red')
        #draw.line((100, 200, 150, 300), fill=500)
    blank_image_size=blank_image.size
    width=blank_image_size[0]
    draw.line((width-x,y,width-x-10,y-10),fill=100,width=5)
    draw.line((width-x,y,width-x-10,y+10),fill=100,width=5)
    draw.line((width-x,y,width-x+10,y-10),fill=100,width=5)
    draw.line((width-x,y,width-x+10,y+10),fill=100,width=5)
    return blank_image
    

# DEFINE PARAMS FOR SELECTING
    

def defineParamsForCircle(params):
    
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    #filter by color
    params.filterByColor=True
    params.blobColor=0 #SVETLE =255
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 4000 
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5  #0.7
    params.maxCircularity = 1     
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2 #0.87
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    return 0
    
    
def defineParamsForTriangle(params):
    
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    #filter by color
    params.filterByColor=True
    params.blobColor=255 #SVETLE
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 2000 
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.maxCircularity = 0.6
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    return 0
     

# SELECT COLOR

def selectColor(image, color):
    
    # create NumPy arrays from the boundaries
    lower = np.array(color[0], dtype = "uint8")
    upper = np.array(color[1], dtype = "uint8")
    
    # find the colors within the specified boundaries and apply
	# the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    #sf.display_image(np.hstack([img, output]))
    #sf.display_image(output)
    return output
    


def selectColorHSV(hue, imgFromCamera):

    lower_range = np.array([max(0, hue - 10), 0, 0], dtype=np.uint8)
    upper_range = np.array([min(180, hue + 10), 255, 255], dtype=np.uint8)
    
    img_hsv = cv2.cvtColor(imgFromCamera, cv2.COLOR_BGR2HSV)    
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    
    binary_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    #_, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    
    return binary_img


def selectColorHSV_v02(hue, imgFromCamera):

    lower_range = np.array([max(0, hue - 10), 0, 0], dtype=np.uint8)
    upper_range = np.array([min(180, hue + 10), 255, 255], dtype=np.uint8)
    
    img_hsv = cv2.cvtColor(imgFromCamera, cv2.COLOR_BGR2HSV)    
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    
    binary_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    #_, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    
    return binary_img







    