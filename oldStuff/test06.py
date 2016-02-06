
import numpy as np
import cv2
from cv2 import *
import softFunctions as sf
import allFunctions as af
import matplotlib.pyplot as plt


#%%

# HOW TO GET HSV FROM RGB
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
#>>> green = np.uint8([[[0,255,0 ]]])
#>>> hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#>>> print hsv_green
#[[[ 60 255 255]]]

green = np.uint8([[[50, 60, 255]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print hsv_green


#%%

params_for_circle = cv2.SimpleBlobDetector_Params()
af.defineParamsForCircle(params_for_circle)
#### Create a detector with the parameters for CIRCLE
detector_for_circle = cv2.SimpleBlobDetector_create(params_for_circle)





frame = sf.load_image("images/aa7.jpg")
#sf.display_image(img)
   
#==================================================
    # Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
#lower_blue = np.array([30,50,50])
#upper_blue = np.array([70,255,255])
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])


    # Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)
#===================================================

res=sf.erode(res)
#res=sf.erode(res)

res, x, y = af.selectBlobFromG_v02(sf.invert(res), params_for_circle, detector_for_circle)  

sf.display_image(res)
    