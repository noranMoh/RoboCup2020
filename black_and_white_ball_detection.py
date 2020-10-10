import cv2
from imutils.video import VideoStream
import time
import imutils
import math


classifier = 'classifier/cascade.xml'

ball_tracker = cv2.CascadeClassifier(classifier)

vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:
	frame = vs.read()


	frame = imutils.resize(frame, width=600)

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	balls = ball_tracker.detectMultiScale(gray_frame,1.1,9)

	
	for (x,y,w,h) in balls:
		
		cv2.circle(frame, (math.ceil(x+w*0.5), math.ceil(y+h*0.5)), math.ceil(w*0.5),
			(255, 0, 255), 2)
		
	
	cv2.imshow('Detect Ball',frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break


cv2.destroyAllWindows()		
