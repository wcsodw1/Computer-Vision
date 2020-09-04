# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:53:02 2020

@author: user
"""

# Python David_4_6_test_model.py

# (Set your own Path with image)python David_4_6_test_model.py --images "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/test_images" --codebook "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/vocab.cpickle" --model "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/model.cpickle"

# (Set your own Path with image)python David_4_6_test_model.py --images "../../data/test_data" --codebook "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/vocab.cpickle" --model "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/model.cpickle"

    # import the necessary packagess
from __future__ import print_function
from SubModule.descriptors import DetectAndDescribe
from SubModule.ir import BagOfVisualWords
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths

import argparse
import pickle
import imutils
import cv2
import sys


    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="Path to input images directory")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the codebook")
ap.add_argument("-m", "--model", required=True,
	help="Path to the classifier")

sys.argv[1:] = '-i test_images -c output/vocab.cpickle -m output/bovw.hdf5 -m output/model.cpickle'.split()
args = vars(ap.parse_args())


# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the classifier
model = pickle.loads(open(args["model"], "rb").read())

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
	# load the image and prepare it from description
	image = cv2.imread(imagePath)
	print("Image : ",image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = imutils.resize(gray, width=min(320, image.shape[1]))

	# describe the image and classify it
	(kps, descs) = dad.describe(gray)
	hist = bovw.describe(descs)
	hist /= hist.sum()
	prediction = model.predict(hist)[0]

	# show the prediction
    
	filename = imagePath[imagePath.rfind("/") + 1:]
	#print("Image : ", image)
	print("Prediction : ", prediction)

	print("[PREDICTION] {}: {}".format(filename, prediction))
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)