import cv2
from imutils.video import VideoStream
import time
import imutils
import math

#load trained classifier 

classifier = 'classifier/cascade.xml'

ball_tracker = cv2.CascadeClassifier(classifier)

#load image to be detected
img_file = "test/23.png"

# create an openCV image
img = cv2.imread(img_file)
# convert color image to grey image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

balls = ball_tracker.detectMultiScale(gray_img)

print(balls)

for (x,y,w,h) in balls:

	#draw a circle around detected balls
	cv2.circle(img, (math.ceil(x+w*0.5), math.ceil(y+h*0.5)), math.ceil(w*0.5),
			(255, 0, 255), 2)


cv2.imwrite(r'test/23_detected.png', img)

cv2.imshow('my detection',img)

# wait for the keystroke to exit
cv2.waitKey()
