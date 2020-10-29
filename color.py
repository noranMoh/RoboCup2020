from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

image_hsv = None

def color (event, x,y,flags,p):
	if event == cv2.EVENT_LBUTTONDOWN:
		pixel = image_hsv[y,x]

		print(np.array([pixel[0],pixel[1],pixel[2]]))

def main():
    import sys
    global image_hsv, pixel # so we can use it in mouse callback

    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    while True:
    	frame= vs.read()

    	cv2.imshow("bgr",frame)
    	cv2.namedWindow('hsv')
    	cv2.setMouseCallback('hsv', color)

   
    	image_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    	cv2.imshow("hsv",image_hsv)

    	key = cv2.waitKey(1) & 0xFF
    	if key == ord("q"):
    		break

    vs.stop()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()