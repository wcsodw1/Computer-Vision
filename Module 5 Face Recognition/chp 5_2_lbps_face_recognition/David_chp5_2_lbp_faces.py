# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:57:05 2020

@author: user
"""
# python David_chp5_2_lbp_faces.py --dataset caltech_faces

# import the necessary packages
from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import argparse
import imutils
import cv2

import sys 
from face_recognition.datasets import load_caltech_faces

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to CALTECH Faces dataset")
ap.add_argument("-s", "--sample-size", type=int, default=21, help="# of example samples")

sys.argv[1:] = '-d ../../../CV_PyImageSearch/Dataset/caltech_faces'.split()
#sys.argv[1:] = '-d image_datasets/caltech_faces'.split()

args = vars(ap.parse_args())

# load the CALTECH faces dataset
print("[INFO] loading CALTECH Faces dataset...")
(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21,
	test_size=0.25)

print("Training : ",training)
print("Testing : ",testing)
print("Names : ",names)

# encode the labels, transforming them from strings into integers since OpenCV does
# not like strings as training data
print("TRPE-LabelEncoder ",type(LabelEncoder))
le = LabelEncoder()
print("le : ", type(le))
le.fit_transform(training.target)
print("le.fit_transform : ", type(le))

# handle if we are creating the LBP face recognizer for OpenCV 2.4
if imutils.is_cv2():
	recognizer = cv2.createLBPHFaceRecognizer(radius=2, neighbors=16, grid_x=8, grid_y=8)

# otherwise, we are creating the LBP face recognizer for OpenCV 3
else:
	recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

# train the Local Binary Pattern face recognizer
print("[INFO] training face recognizer...")
recognizer.train(training.data, le.transform(training.target))

# initialize the list of predictions and confidence scores
print("[INFO] gathering predictions...")
predictions = []
confidence = []

# loop over the test data
for i in range(0, len(testing.data)):
	print("{} of {}".format(str(i), str(len(testing.data))))
	# classify the face and update the list of predictions and confidence scores
	(prediction, conf) = recognizer.predict(testing.data[i])
	print("prediction : ", prediction) # The value of LBP
	print("conf : ", conf) # conf : prediction class 
	predictions.append(prediction)
	confidence.append(conf)

    # show the classification report
print("Classification Report: ",classification_report(le.transform(testing.target), predictions,
	target_names=np.unique(names)))

# loop over the the desired number of samples
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
	# resize the face to make it more visable, then display the face and the prediction
	print("[INFO] Prediction: {}, Actual: {}, Confidence: {:.2f}".format(
		le.inverse_transform([predictions[i]])[0], testing.target[i], confidence[i]))
	face = testing.data[i]
   #face = imutils.resize(face, width=face.shape[1] * 2, inter=cv2.INTER_CUBIC)
	face = cv2.resize(face,(400,600), interpolation=cv2.INTER_CUBIC)
	cv2.imshow("Face", face)
	#cv2.waitKey(0)
	cv2.waitKey(1000)
	cv2.destroyAllWindows()
