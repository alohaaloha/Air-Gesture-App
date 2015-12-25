# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:57:03 2015

@author: aloha
"""

import softFunctions as sf
import numpy as np
import cv2
from PIL import Image, ImageDraw
import numpy as np

    
    
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
    
    
#---------------------
    
    
def drawDotOnImg(blank_image, x, y, width):
    draw = ImageDraw.Draw(blank_image)
    draw.ellipse((width-x, y, width-x+10, y+10), fill = 'red', outline ='red')
        #draw.line((100, 200, 150, 300), fill=500)
    
    
def connectDots(blank_image, x1, y1, x2, y2, width):
    draw = ImageDraw.Draw(blank_image)
    draw.line((width-x1, y1, width-x2, y2), fill=500, width=10)
    
    
#--------------------
    
    
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
    
    
    
def defineParamsForCircle(params):
    
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
    params.minCircularity = 0.7
    params.maxCircularity = 1     
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    
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
     
    
    