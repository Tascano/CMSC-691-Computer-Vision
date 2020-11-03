# -*- coding: utf-8 -*-
"""

Converts RGB image capture in camera to 160x90 image in binary using otsu's threshold'

"""

import cv2
import os

#Load image and rezie the dimensions to 160x90
path = os.getcwd()
print(path)

folder = os.path.join(path,r'Final Dataset\Nishan\RGB\Fist')
print(path, folder)
count = 0

for filename in os.listdir(folder):
    print(filename)
    img = cv2.imread((os.path.join(folder, filename)),0)
    #cv2.imshow("Image",image)
    y = 10
    x = 60
    h = 640
    w = 420
    image = img[x:w, y:h]
    #image = img
    image = cv2.medianBlur(image,15)
    #print(img.shape)
    width = 112
    height = 63
    
    dsize = (width, height)
    r_image = cv2.resize(image, dsize)
    
    #print(resized_image.shape)
    #cv2.imshow("Resized Image", resized_image)
    
    #Convert to grayscale and use otsu's threshold
    #ret,th1 = cv2.threshold(r_image,127,255,cv2.THRESH_BINARY_INV)

    th2 = cv2.adaptiveThreshold(r_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    #cv2.imshow("Th", thresh)
    output_path_gaussian = "D:/UMBC/Sem 2/CV/Term Project/Final Dataset/Nishan/Gaussian/Fist/Fist_{}.jpg".format(count)
    #output_path_global = "D:/UMBC/Sem 2/CV/Term Project/Parthan/Global/TU/TU_{}.jpg".format(count)
    #cv2.imshow("Binary", th)
    cv2.imwrite(output_path_gaussian, th2)
    #cv2.imwrite(output_path_global, th1)
    print("Gaussian {} written".format(count))
    #print("Binary {} written".format(count))
    count += 1
    cv2.waitKey()
    