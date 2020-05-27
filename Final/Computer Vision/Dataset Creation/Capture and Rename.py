# -*- coding: utf-8 -*-
"""
Parthan:
    
DataSet creation:
    
Capture images and Rename folders and image files

"""

import os
import cv2
import time

def renameFolder():
    """

    To rename the folder where the image is captured
    
    """
    for count, filename in enumerate(os.listdir("D:/UMBC/Sem 2/CV/Term Project/Hand Gesture Dataset/Thumbs Up")):
        dst = "Thumbs_Up_" + str(count) + ".jpg"
        src = "D:/UMBC/Sem 2/CV/Term Project/Hand Gesture Dataset/Thumbs Up/" + filename
        dst = "D:/UMBC/Sem 2/CV/Term Project/Hand Gesture Dataset/Thumbs Up/" + dst
        
        os.rename(src, dst)
        
def captureImages():
    capture = cv2.VideoCapture(0)
    img_counter = 0
    start_time = time.time()
    
    while True:
        ret,image = capture.read()
        cv2.imshow('Image', image)
        
        #Press q to exit out of the camera capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= 3: # Time difference of 3 seconds
            img_name = "D:/UMBC/Sem 2/CV/Term Project/Testing/test_{}.jpg".format(img_counter)  #Directory where image can be captured and written to
            cv2.imwrite(img_name, image)
            print("Image {} written".format(img_counter))   # Identify the captured image
            start_time = time.time()
            img_counter += 1
        
    capture.release()
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    captureImages()
    #renameFolder()