# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:35:03 2020

@author: Laila
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

lookup = dict()
reverselookup = dict()
count = 0

for j in os.listdir("D:/UMBC/Sem 2/CV/Term Project/leapGestRecog/00"):
    if not j.startswith("."):
        lookup[j] = count
        reverselookup[count] = j
        count += 1
        
#print(lookup)

x_data = []
y_data = []
datacount = 0
for i in range(0, 10):
    for j in os.listdir('D:/UMBC/Sem 2/CV/Term Project/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0
            for k in os.listdir('D:/UMBC/Sem 2/CV/Term Project/leapGestRecog/0' + str(i) + '/' + j + '/'):
                img = Image.open('D:/UMBC/Sem 2/CV/Term Project/leapGestRecog/0' + str(i) + '/' + j + '/' + k).convert('L')
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

print(x_data)

from random import randint
for i in range(0, 10):
    plt.imshow(x_data[i*200 , :, :])
    plt.title(reverselookup[y_data[i*200 ,0]])
    plt.show()
