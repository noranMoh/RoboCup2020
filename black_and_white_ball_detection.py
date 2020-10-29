import cv2
from imutils.video import VideoStream
import time
import imutils
import math

#to save video
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('balck_and_white_ball_tracking.avi', fourcc, 30.0, (640,480))

classifier = 'classifier/cascade.xml'

ball_tracker = cv2.CascadeClassifier(classifier)

#start video cam
vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:

	#read frame
	frame = vs.read()

	#turn frame to gray scale
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	balls = ball_tracker.detectMultiScale(gray_frame,1.1,9)

	
	for (x,y,w,h) in balls:
		
		cv2.circle(frame, (math.ceil(x+w*0.5), math.ceil(y+h*0.5)), math.ceil(w*0.5),
			(255, 0, 255), 2)
		
	#save frame
	out.write(frame)
	
	#show frame
	cv2.imshow('Detect Ball',frame)
	
	#break when q is pressed
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

out.release()
cv2.destroyAllWindows()		
