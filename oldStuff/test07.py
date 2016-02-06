# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:59:01 2015

@author: aloha
"""

 
 #%% IMPORT LIBRARIES
 
#import potrebnih biblioteka za K-means algoritam
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
# iris = datasets.load_iris() #Iris dataset koji će se koristiti kao primer https://en.wikipedia.org/wiki/Iris_flower_data_set

#import potrebnih biblioteka
import cv2
import collections

import softFunctions as sf
import allFunctions as af

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab

#%%%
#Funkcionalnost implementirana u V2
def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
    
    
      

def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(26, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=500, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False) 
      
    return ann



# vrati prepoznato u obliku liste
def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


    

#%%
#*****************************************************************************
#***** 1)LOAD - 2)PREPARE_IMG - 3)SELECT_ROI - 4)LEARN - 5)TEST_1 ************
#***************************************************************************** 


#%% 1) LOAD
image_color = sf.load_image("images/alphabet.png")
sf.display_image(image_color)

#%% 2) PREPARE_IMG
img = sf.image_bin(sf.image_gray(image_color))

sf.display_image(img)

#%% 3) SELECT_ROI
selected_regions, letters_regions, region_distances = sf.select_roi(image_color.copy(), img)
sf.display_image(selected_regions) #selected_regions je slika sa oznacenim regionima
print 'Broj prepoznatih regiona:', len(letters_regions)
#print region_distances

#%% 4) LEARN
alphabet = ['A','B','C','D','E','F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

inputs = prepare_for_ann(letters_regions)
outputs = convert_output(alphabet)

ann = create_ann()
ann = train_ann(ann, inputs, outputs)

#%% 5) TEST_1
# testing if ann can even predict stuff used for learning
result = ann.predict(np.array(inputs, np.float32)) 
#print result # vrednosti izlaza (od 0 do 1 - ono sa table)
sf.display_image(selected_regions) # slika sa oznacenim regionima
print 'RESENJE: '
print display_result(result, alphabet)

# E _ N _ D


#%%
#*****************************************************************************
#***** 1)LOAD - 2)PREPARE_IMG - 3)SELECT_ROI - 4)K-MEANS - 5)DISPLAY 6)PREDICT************
#***************************************************************************** 

#%% 1) LOAD
image_color = sf.load_image("images/timg.jpg")
sf.display_image(image_color)

#%% 2) PREPARE_IMG
img = sf.image_bin(sf.image_gray(image_color))
img = sf.invert(img)
img = sf.erode(img)
img = sf.erode(img)
img=sf.dilate(img)
img=sf.dilate(img)
img=sf.dilate(img)
img=sf.dilate(img)
sf.display_image(img)

#%%
test = image_bin(image_gray(image_color))
test_bin = dilate(test)
test_bin = erode(test_bin)

display_image(test_bin)

img = invert(test_bin)
img=dilate(img)
display_image(img)

#%% 3) SELECT_ROI
selected_regions, letters, distances = select_roi_better(image_color.copy(), img)
display_image(selected_regions)
print 'Broj prepoznatih regiona:', len(letters)
print distances

#%% 4) K-MEANS
#Podešavanje centara grupa K-means algoritmom
distances = np.array(distances).reshape(len(distances), 1)
#Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
k_means.fit(distances)

#%% 5) DISPLAY
def display_result_k(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    print w_space_group
    result = ""#alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[0:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += 'GORE:'
        result += alphabet[winner(output)]
    return result

#%% 6) PREDICT
inputs = prepare_for_ann(letters)
results = ann.predict(np.array(inputs, np.float32))
#print display_result(results, alphabet, k_means)
print display_result(results, alphabet)

rez=display_result(results, alphabet)
#print "Trouglova ima: ", rez.count("Trougao")
#print "Kvadrata ima: ", rez.count("Kvadrat")

#%% 7) K-MEANS
print display_result_k(results, alphabet, k_means)
recenica=display_result_k(results, alphabet, k_means);
#print recenica
