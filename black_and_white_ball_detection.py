import cv2


cap = cv2.VideoCapture(0)

classifier = 'classifier/cascade.xml'

ball_tracker = cv2.CascadeClassifier(classifier)

while True:
	frame = cap.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	ball = ball_tracker.detectMultiScale(gray_frame,1.1,9)

	cv2.imshow('Detect Ball',frame)