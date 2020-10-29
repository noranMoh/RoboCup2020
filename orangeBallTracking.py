from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time



def main():

	#define the lower and upper bound of the color

	#orangeLower = (0, 70, 185)
	#orangeUpper = (232, 255, 255)

	orangeLower = (0, 100, 180)
	orangeUpper = (30, 200, 255)

	#start webcam
	vs = VideoStream(src=0).start()

	# to save the video
	fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
	out = cv2.VideoWriter('orange_ball_tracking.avi', fourcc, 30.0, (600,450))


	# wait for 2 seconds
	time.sleep(2.0)

	while True:

		# get current frame
		frame = vs.read()
		
		#resize frame (for faster processing)

		frame = imutils.resize(frame, width=600)

		# blur frame (to reduce noise)

		blurred = cv2.GaussianBlur(frame, (11, 11), 0)

		# convert to HSV color space

		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		#create a mask on the orange color using pre-set lower and upper bounds
		mask = cv2.inRange(hsv, orangeLower, orangeUpper)

		# erode and dilate to remove small blobs left in the mask	
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask 
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)

		cnts = imutils.grab_contours(cnts)

		#intialize ball centre
		center = None

		# check if at least one contour was found
		if len(cnts) > 0:

			# find the largest contour in the mask
			c = max(cnts, key=cv2.contourArea)

			((x, y), radius) = cv2.minEnclosingCircle(c)
			
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# draw the circle and centre on the frame
			cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

		#save frame
		out.write(frame)

		#show frame
		cv2.imshow("Detection", frame)

		#break when q is pressed
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	#release output video
	out.release()

	#stop video stream
	vs.stop()

	cv2.destroyAllWindows()

if __name__=='__main__':
    main()	