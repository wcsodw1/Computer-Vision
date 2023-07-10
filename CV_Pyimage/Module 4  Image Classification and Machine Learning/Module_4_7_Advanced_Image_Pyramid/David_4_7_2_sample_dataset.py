# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:36:38 2020

@author: user
"""
# This Chapter is going to make our sample directory 
# split training and testing data into folder 

# python David_4_7_2_sample_dataset.py --input ~/PyImageSearch/Datasets/caltech5 --output output/data --training-size 0.75
# python David_4_7_2_sample_dataset.py --input "../../../CV_PyImageSearch/Dataset/caltech5" --output output/data --training-size 0.75
# python David_4_7_2_sample_dataset.py

# 1.Preprocessing
    
    # 1.1 import the necessary packages
from imutils import paths
import random
import shutil

import argparse
import glob
import os
import sys

    # 1.2 construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of image classes")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store training and testing images")
ap.add_argument("-t", "--training-size", type=float, default=0.75,
	help="% of images to use for training data")

sys.argv[1:] = '-i ../../../CV_PyImageSearch/Dataset/caltech5 -o output/data -t 0.75'.split()
# sys.argv[1:] = '-i ../../../CV_PyImageSearch/Dataset/caltech5 -o ../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_7_AdvancedImagePyramid/output/data -t 0.75'.split()

args = vars(ap.parse_args())


    # 1.3 (!!)if the output directory exists, delete it
if os.path.exists(args["output"]):
	shutil.rmtree(args["output"]) # shutil.rmtree 刪除目錄和下面的所有檔案 使用
    
    # 1.4 (!!) create the output directories : 創建自己想要的檔案及路徑
os.makedirs(args["output"])
os.makedirs("{}/training".format(args["output"]))
os.makedirs("{}/testing".format(args["output"]))


# 2.loop over the image classies in the input directory
for labelPath in glob.glob(args["input"] + "\\*"):
	# extract the label from the path and create the sub-directories for the label in
	# the output directory
	label = labelPath[labelPath.rfind("\\") + 1:] # File caltech5 往後數一個資料夾中裡面的資料 
	print(label)
	os.makedirs("{}/training/{}".format(args["output"], label))
	os.makedirs("{}/testing/{}".format(args["output"], label))

	# grab the image paths for the current label and shuffle them
	imagePaths = list(paths.list_images(labelPath))
	print(imagePaths) # Grab all the image in the folder and save to "labelPath"
	random.shuffle(imagePaths)
	i = int(len(imagePaths) * args["training_size"])
	print(i)
	# loop over the randomly sampled training paths and copy them into the appropriate
	# output directory
	for imagePath in imagePaths[:i]:
		filename = imagePath[imagePath.rfind("\\") + 1:]
		shutil.copy(imagePath, "{}/training/{}/{}".format(args["output"], label, filename))

	# loop over the randomly sampled testing paths and copy them into the appropriate
	# output directory
	for imagePath in imagePaths[i:]:
		filename = imagePath[imagePath.rfind("\\") + 1:]
		shutil.copy(imagePath, "{}/testing/{}/{}".format(args["output"], label, filename))
