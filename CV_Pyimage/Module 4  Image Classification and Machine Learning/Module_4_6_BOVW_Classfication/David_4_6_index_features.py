# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:50:58 2020

@author: user
"""

# USAGE

# python David_4_6_index_features.py --dataset "../../../CV_PyImageSearch/Dataset/caltech5" --features-db "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/features.hdf5" --approx-images 500

# python python David_4_6_index_features.py --dataset caltech5 --features-db output/features.hdf5 --approx-images 500

# 1.Preprocessing : 
    # 1.1 import the necessary packages
from __future__ import print_function
from SubModule.descriptors import DetectAndDescribe
from SubModule.indexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import random
import cv2

import sys

    # 1.2 Parsed the argument : 
    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True,
	help="Path to where the features database will be stored")
ap.add_argument("-a", "--approx-images", type=int, default = 500,
	help="Approximate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
	help="Maximum buffer size for # of features to be stored in memory")

# sys.argv[1:] = '-d caltech5 -f output/features.hdf5'.split()
args = vars(ap.parse_args())


# 2. 
    # 2.1 
    # initialize the 1.keypoint detector 2.local invariant descriptor,
    # and the 3. descriptor pipeline
detector = FeatureDetector_create("GFTT")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)
    # approx_images :  int , 500
    # dataset : str ,  ../../caltech5
    # features_db : str ,   output/features.hdf5
    # max_buffer_size : int, 50000
    
    # 2.2 
    # 1.initialize the feature indexer
    # 2.then grab the image paths 
    # 3.randomly shuffle them
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],maxBufferSize=args["max_buffer_size"], verbose=True) 
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths) # Shuffle the arrange


# 3. loop over the images in the dataset
    # 3.1 Grab all image
    # loop over the images in the dataset
for (i, imagePath) in enumerate(imagePaths):
   
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# extract the filename and image class from the image path and use it to
	# construct the unique image ID
    #print(imagePath)
	p = imagePath.split("\\") # In windows : Change (/) to (\\) 
	print(p)
	imageID = "{}:{}".format(p[-2], p[-1])
	print(imageID)
	#id1 = imagePath.find(".")
	#imageID = imagePath[19:-4]
	#imageID = imagePath[id1+6:]
   
	# 3.2 Turn to Grayscale (3D->2D)
   # load the image and prepare it from description
	image = cv2.imread(imagePath)
	print(image)
	print(image.shape)
	image = imutils.resize(image, width=min(320, image.shape[1]))
	print(image)
	print(image.shape)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Change to 2 Dimension
	print(image)
	print(image.shape)

	# describe the image
	(kps, descs) = dad.describe(image)
	print(kps, descs)
	# if either the keypoints or descriptors are None, then ignore the image
	if kps is None or descs is None:
		continue

	# index the features
	fi.add(imageID, kps, descs)

# finish the indexing process
fi.finish()



