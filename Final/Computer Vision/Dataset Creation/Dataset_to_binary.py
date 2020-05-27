# -*- coding: utf-8 -*-
"""

Converts RGB image capture in camera to 160x90 image in binary using otsu's threshold'

"""

import cv2
import os

#Load image and rezie the dimensions to 160x90
path = os.getcwd()
#print(path)
folder = os.path.join(path,r'Hand Gesture Dataset\Palm')
print(path, folder)
count = 0

for filename in os.listdir(folder):
    #print(filename)
    image = cv2.imread((os.path.join(folder, filename)))
    #cv2.imshow("Image",image)
    print(image.shape)
    width = 160
    height = 90
    
    dsize = (width, height)
    resized_image = cv2.resize(image, dsize)
    
    print(resized_image.shape)
    
    #Convert to grayscale and use otsu's threshold
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    output_path = "D:/UMBC/Sem 2/CV/Term Project/Binary/Palm/Palm_{}.jpg".format(count)
    #cv2.imshow("Binary", thresh)
    cv2.imwrite(output_path, thresh)
    print("Binary {} written".format(count))
    count += 1
    cv2.waitKey()
    