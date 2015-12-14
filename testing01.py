# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:52:48 2015

@author: aloha
"""

import softFunctions as sf
import allFunctions as af
import cv2
from PIL import Image, ImageDraw
import webbrowser
import numpy as np

#%%
image = Image.open("blob1.jpg")
draw = ImageDraw.Draw(image)
draw.ellipse((20, 20, 100, 100), fill = 'blue', outline ='blue')
sf.display_image(image)
#image.show()
cv2.imshow("APP", np.asarray(image.convert('L')))

im = Image.new('RGBA', (400, 400), (0, 255, 0, 0))
cv2.imshow("adas", im)
raw_input("Press Enter to continue...")