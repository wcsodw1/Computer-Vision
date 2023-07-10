# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:47:29 2020

@author: user
"""
# python David_4_7_2_test_model.py --images "output/data/testing" --codebook "output/vocab.cpickle" --levels 2 --model "output/model_L3.cpickle"
# python David_4_7_2_test_model.py

# import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.descriptors import PBOW
from pyimagesearch.ir import BagOfVisualWords
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import numpy as np
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
ap.add_argument("-l", "--levels", type=int, default=2,
	help="# of pyramid levels to generate")
ap.add_argument("-m", "--model", required=True,
	help="Path to the classifier")

sys.argv[1:] = '-i output/data/testing -c output/vocab.cpickle -l 4 -m output/model_L4.cpickle'.split()
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
# and the pyramid of bag-of-visual-words descriptor
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)
pbow = PBOW(bovw, numLevels=args["levels"])

# load the classifier and grab the list of image paths
model = pickle.loads(open(args["model"], "rb").read())
imagePaths = list(paths.list_images(args["images"]))

# initialize the list of true labels and predicted labels
print("[INFO] extracting features from testing data...")
trueLabels = []
predictedLabels = []

i=0

# loop over the image paths
for imagePath in imagePaths:
	# extract the true label from the image patn and update the true labels list	
   Label1=imagePath.split("\\")[1] # Ex : motorbike
   print(Label1)   
   trueLabels.append(Label1)
   print("i=",i,"imagePath=",imagePath,"Label[1]=", Label1)
   i=i+1
   # trueLabels.append(imagePath.split("/")[-2])

	# load the image and prepare it from description
   image = cv2.imread(imagePath)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = imutils.resize(gray, width=min(320, image.shape[1]))

	# describe the image and classify it
   (kps, descs) = dad.describe(gray)
   hist = pbow.describe(gray.shape[1], gray.shape[0], kps, descs)
   prediction = model.predict(hist)[0]
   predictedLabels.append(prediction)

# show a classification report
print(classification_report(trueLabels, predictedLabels))

for i in np.random.choice(np.arange(0, len(imagePaths)), size=(30,), replace=False):
	# load the image and show the prediction
	image = cv2.imread(imagePaths[i])

	# show the prediction
	filename = imagePaths[i][imagePaths[i].rfind("/") + 1:]
	print("filename",filename)
	#print(predictedLabels)
	print("[PREDICTION] {}: {}".format(filename, predictedLabels[i]))
	cv2.putText(image, predictedLabels[i], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 2)
	image = cv2.resize(image, (500,700), interpolation = cv2.INTER_CUBIC)
	cv2.imshow("Image", image)
	cv2.waitKey(2000)
	cv2.destroyAllWindows()
    
print(classification_report(trueLabels, predictedLabels))

