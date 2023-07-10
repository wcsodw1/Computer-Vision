import numpy as np
import cv2
 
face_cascade = cv2.CascadeClassifier("C:/Users/user/Desktop/AI with Computer Vision/Computer-Vision/Github/Module 5 Face Recognition/chp 5_4_gathering_selfies/cascades/haarcascade.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/user/Desktop/AI with Computer Vision/Computer-Vision/Github/Module 5 Face Recognition/chp 5_4_gathering_selfies/cascades/haarcascade_eye.xml")
 
img = cv2.imread("2.JPG")
img = cv2.resize(img,(400,600),interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    
faces = face_cascade.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))
 
if len(faces)>0:
    for faceRect in faces:
        x,y,w,h = faceRect
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)
 
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
 
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(2,2))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------

'''# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:25:30 2020

@author: user
"""

# Different Classifier Download : https://github.com/opencv/opencv/tree/master/data/haarcascades

# python David_chp5_4_gather_selfies.py --face-cascade "cascades/haarcascade_frontalface_default.xml" --output "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_5_4_face detetion/output/faces/david.txt"
# python David_chp5_4_gather_selfies.py

    # import the necessary packages
from __future__ import print_function
from imutils import encodings
from pyimagesearch.face_recognition import FaceDetector

import argparse
import imutils
import cv2
import sys

    # construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-e", "--eye-cascade", required=True, help="path to eye detection cascade")

ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")


sys.argv[1:] = '-f cascades/haarcascade.xml -e cascades/haarcascade_eye.xml -o ../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_5_4_face_detetion/output/faces/David_data.txt'.split() # 
args = vars(ap.parse_args())


    # 2.1 initialize the face detector, boolean indicating if we 
    # are in capturing mode or not, and　the bounding box color
    # 2.2 cv2.CascadeClassifier :https://blog.csdn.net/GAN_player/article/details/77993872

fd = FaceDetector(args["face_cascade"])
ed = FaceDetector(args["eye_cascade"])

captureMode = False
color = (0, 255, 0)

# grab a reference to the webcam and open the output file for writing
camera = cv2.VideoCapture(0)
f = open(args["output"], args["write_mode"])
total = 0



# loop over the frames of the video
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end of the video
	if not grabbed:
		break

	# resize the frame, convert the frame to grayscale, and detect faces in the frame
	frame = imutils.resize(frame, width = 500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100))
	eyedetect = ed.detect(gray, scaleFactor=1.1, minNeighbors=9, minSize=(20, 20))
    
    #eye detect
# 	img = cv2.imread("2.JPG")
	img = imutils.resize(frame, width = 500)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# 	faces = face_cascade.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))
    
	print(faceRects)
	# ensure that at least one face was detected
	if len(faceRects) > 0:
		# sort the bounding boxes, keeping only the largest one
		(x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))

		# if we are in capture mode, extract the face ROI, encode it, and write it to file
		if captureMode:
			face = gray[y:y + h, x:x + w].copy(order="C")
			f.write("{}\n".format(encodings.base64_encode_image(face)))
			total += 1

	if len(eyedetect)>0:
		for faceRect in eyedetect:
			x,y,w,h = faceRect
# 			f.write("{}\n".format(encodings.base64_encode_image(face)))
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)
			roi_gray = gray[y:y+h,x:x+w]
			roi_color = img[y:y+h,x:x+w]

			eyes = ed.detectMultiScale(roi_gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(2,2))
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)            
            
		# draw bounding box on the frame
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
		cv2.circle(frame, (x+100, y+100), 100, (0, 0, 255), 2)

	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `c` key is pressed, then go into capture mode
	if key == ord("c"):
		# if we are not already in capture mode, drop into capture mode
		if not captureMode:
			captureMode = True
			color = (0, 200, 255)

		# otherwise, back out of capture mode
		else:
			captureMode = False
			color = (255, 200, 0)

	# if the `q` key is pressed, break from the loop
	elif key == ord("q"):
		break
    
# close the output file, cleanup the camera, and close any open windows
print("[INFO] wrote {} frames to file".format(total))
f.close()
camera.release()'''