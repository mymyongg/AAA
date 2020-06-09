import cv2
import numpy as numpy

def add_blur(image, x,y,hw):
    image[y:y+hw, x:x+hw,1] = image[y:y+hw, x:x+hw,1]+30
    image[:,:,1][image[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image[y:y+hw, x:x+hw,1] = cv2.blur(image[y:y+hw, x:x+hw,1] ,(10,10))
    return image

image = cv2.imread('dataset/images/30/00005.png')
foggy = add_blur(image, 0, 0, 160)
cv2.imshow('foggy', foggy)
cv2.waitKey(0)