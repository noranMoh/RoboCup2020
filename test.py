from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time



frameWidth = 640
framHeight = 480

cap = cv2.VideoCapture(0)

cap.set(3,frameWidth)
cap.set(4,framHeight)

while True:
    success, img = cap.read()
    #img = cv2.resize(img,(frameWidth,framHeight))
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    cv2.imshow('Original',img)
    cv2.imshow("HSV",imgHSV)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
