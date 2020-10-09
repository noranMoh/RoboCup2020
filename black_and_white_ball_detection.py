import cv2
from imutils.video import VideoStream
import time
import imutils
import math


classifier = 'classifier/cascade.xml'

#classifier = 'classifier/ball_cascade.xml'

ball_tracker = cv2.CascadeClassifier(classifier)


img_file = "test/log_43603.png"

# create an openCV image
img = cv2.imread(img_file)
# convert color image to grey image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

balls = ball_tracker.detectMultiScale(gray_img)

print(balls)

for (x,y,w,h) in balls:

	cv2.circle(img, (math.ceil(x+w*0.5), math.ceil(y+h*0.5)), math.ceil(w*0.5),
			(0, 255, 255), 2)

cv2.imshow('my detection',img)

# wait for the keystroke to exit
cv2.waitKey()


"""
vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:
	frame = vs.read()
	#frame = frame[1]

	frame = imutils.resize(frame, width=600)

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	balls = ball_tracker.detectMultiScale(gray_frame,1.1,9)

	
	for (x,y,w,h) in balls:

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
		#cv2.circle(frame, (int(x), int(y)), int(w*0.5),
		#		(0, 255, 255), 2)
		#cv2.circle(frame, center, 5, (0, 0, 255), -1)
	
	cv2.imshow('Detect Ball',frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break


cv2.destroyAllWindows()		
"""