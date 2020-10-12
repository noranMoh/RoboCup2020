from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

# define the lower and upper boundaries of the "orange"
# ball in the HSV color space, then initialize the
# list of tracked points

orangeLower = (0, 70, 185)
orangeUpper = (232, 255, 255)


#start webcam
vs = VideoStream(src=0).start()

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('orange_ball_tracking.avi', fourcc, 30.0, (600,450))


# wait for 2 seconds
time.sleep(2.0)

while True:

	# get the current frame
	frame = vs.read()
	
	# resize the frame, blur it, and convert it to the HSV
	# color space
	
	frame = imutils.resize(frame, width=600)
	(w, h, c) = frame.shape

	print(w,h)	


	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "orange", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask

	mask = cv2.inRange(hsv, orangeLower, orangeUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# check if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# check if the radius meets a minimum size
		if radius > 20:
			# draw the circle and centroid on the frame
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# show the frame to our screen
	out.write(frame)
	cv2.imshow("Detection", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

out.release()
#stop video stream
vs.stop()

cv2.destroyAllWindows()