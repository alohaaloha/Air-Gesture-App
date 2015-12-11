# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:15:21 2015

TODO
nasi uslovi

@author: aloha
"""

import softFunctions as sf
# Standard imports
import cv2
import numpy as np;
 
# Read image
im = cv2.imread("blob1.jpg", cv2.IMREAD_GRAYSCALE)
 
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