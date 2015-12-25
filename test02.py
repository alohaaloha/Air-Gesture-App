# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 04:07:42 2015

@author: aloha
"""

import softFunctions as sf
# Standard imports
import cv2
import numpy as np;
 
#%%
# Read image
im = cv2.imread("images/blob4.jpg", cv2.IMREAD_GRAYSCALE)

im=sf.invert(im)

sf.display_image(im)
#im=sf.invert(im)

#%% 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
 
# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
 
sf.display_image(im_with_keypoints)
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)