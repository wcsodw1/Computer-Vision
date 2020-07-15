# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:40:07 2020

@author: user
"""
# python David_4_6_extract_bovw.py --features-db "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/features.hdf5" --codebook "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/vocab.cpickle" --bovw-db "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/bovw.hdf5"

# 1.Preprocessing

    # 1.1 import the necessary packages
from SubModule.ir import BagOfVisualWords
from SubModule.indexer import BOVWIndexer
import h5py
import pickle
import argparse
import sys

    # 1.2 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
	help="Path the features database")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the codebook")
ap.add_argument("-b", "--bovw-db", required=True,
	help="Path to where the bag-of-visual-words database will be stored")
#ap.add_argument("-s", "--max-buffer-size", type=int, default=500,
#	help="Maximum buffer size for # of features to be stored in memory")
ap.add_argument("-s", "--max-buffer-size", type=int, default=500,
	help="Maximum buffer size for # of features to be stored in memory")
    
# sys.argv[1:] = '-f output/features.hdf5 -c output/vocab.cpickle -b output/bovw.hdf5'.split()
args = vars(ap.parse_args())

# 2 :
    # 2.1 :   
    # 1.load the codebook vocabulary 
    # 2.initialize the bag-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read()) # (512, 128)
bovw = BagOfVisualWords(vocab)

    # 2.2 : 
    # 1. open the features database(feature.hdf5)
    # 2. initialize the bag-of-visual-words indexer
featuresDB = h5py.File(args["features_db"], mode="r")
bi = BOVWIndexer(bovw.codebook.shape[0], args["bovw_db"],
	estNumImages=featuresDB["image_ids"].shape[0],
	maxBufferSize=args["max_buffer_size"])


# 3. loop over the image IDs and index
for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		bi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# extract the feature vectors for the current image using the starting and
	# ending offsets (while ignoring the keypoints) and then quantize the
	# features to construct the bag-of-visual-words histogram
	features = featuresDB["features"][offset[0]:offset[1]][:, 2:]
	print(features)
	hist = bovw.describe(features)
	print(hist)

	# normalize the histogram such that it sums to one then add the
	# bag-of-visual-words to the index
	hist /= hist.sum()
	bi.add(hist)

# 4. 
    # 1.close the features database
    # 2.finish the indexing process
featuresDB.close()
bi.finish()
