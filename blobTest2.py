# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 04:07:42 2015

@author: aloha
"""
import softFunctions as sf
import allFunctions as af
import cv2
from PIL import Image, ImageDraw
import webbrowser
import numpy as np

#%%
im = cv2.imread("blbT.jpg", cv2.IMREAD_GRAYSCALE)
sf.display_image(im)
cv2.imshow("app", im)

#%%
image = Image.open("blob1.jpg")
draw = ImageDraw.Draw(image)
draw.ellipse((20, 20, 100, 100), fill = 'blue', outline ='blue')
sf.display_image(image)
#image.show()
cv2.imshow("APP", np.asarray(image.convert('L')))




#%%
#from matplotlib import pyplot as plt
#plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#cv2.imshow("", image)
#webbrowser.open("blobT.jpg")