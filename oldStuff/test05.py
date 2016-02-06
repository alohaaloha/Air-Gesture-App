
import numpy as np
import cv2
from cv2 import *
import softFunctions as sf
import allFunctions as af
import matplotlib.pyplot as plt


#%%

params_for_circle = cv2.SimpleBlobDetector_Params()
af.defineParamsForCircle(params_for_circle)
#### Create a detector with the parameters for CIRCLE
detector_for_circle = cv2.SimpleBlobDetector_create(params_for_circle)


img = sf.load_image("images/blob2.jpg")
print img.size
sf.display_image(img)

#%%

#bin_img=cv2.GaussianBlur(bin_img, (5,5),0) 
bin_img=af.selectColorHSV(80, img)

bin_img=sf.erode(bin_img)
bin_img=sf.erode(bin_img)

for i in range(0,10):
    bin_img=sf.dilate(bin_img)



sf.display_image(bin_img)   



#%%   
   
frameWithCircle, x, y = af.selectBlobFromG_v02(sf.invert(bin_img), params_for_circle, detector_for_circle)  
sf.display_image(frameWithCircle)  
   
   