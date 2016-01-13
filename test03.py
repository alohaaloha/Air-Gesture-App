# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 04:07:42 2015
#INFO
# http://docs.opencv.org/master/d0/d7a/classcv_1_1SimpleBlobDetector.html#gsc.tab=0
# http://www.learnopencv.com/blob-detection-using-opencv-python-c/ <<--- ful yea ovo ono
@author: aloha
"""

import numpy as np
import cv2
import softFunctions as sf

#%%
im = cv2.imread("images/fa.png", cv2.IMREAD_GRAYSCALE)
#im=sf.invert(im)
sf.display_image(im)

#%%
import cv2
import numpy as np;

# Read image
im = cv2.imread("images/gg2.jpg", cv2.IMREAD_GRAYSCALE)
#im=sf.invert(im)
#%%
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
    # Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
    
    #filter by color
params.filterByColor=True
params.blobColor=0 #SVETLE 255
    
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

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
sf.display_image(im_with_keypoints)
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)

#print dir(params)
