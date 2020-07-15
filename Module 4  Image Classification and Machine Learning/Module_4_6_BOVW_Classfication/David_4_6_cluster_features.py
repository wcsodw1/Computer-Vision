# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:35:33 2020

@author: user
"""

# python David_4_6_cluster_features.py --features-db "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/features.hdf5" --codebook "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp_4_6_BOVW_Classification/vocab.cpickle" --clusters 512 --percentage 0.25

# python David_4_6_cluster_features.py --features-db output/features.hdf5 --codebook output/vocab.cpickle --clusters 512 --percentage 0.25


# 1.Preprocessing

    # 1.1 import the necessary packages
from __future__ import print_function
from SubModule.ir import Vocabulary
import argparse
import pickle
import sys

    # 1.2 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
	help="Path to where the features database will be stored")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the output codebook")
#ap.add_argument("-k", "--clusters", type=int, default=64,
#	help="# of clusters to generate")
ap.add_argument("-k", "--clusters", type=int, default=512,
	help="# of clusters to generate")
ap.add_argument("-p", "--percentage", type=float, default=0.25,
	help="Percentage of total features to use when clustering")

# sys.argv[1:] = '-f output/features.hdf5 -c output/vocab.cpickle'.split()
    
    # Can visualize the parameter that we declare(宣告) in Spyder 
    # 4 Argument that we parsed 
args = vars(ap.parse_args())  


# 2. (!!!)create the visual words vocabulary
voc = Vocabulary(args["features_db"])
vocab = voc.fit(args["clusters"], args["percentage"])


# 3. dump the clusters to file
print("[INFO] storing cluster centers...")
f = open(args["codebook"], "wb")
f.write(pickle.dumps(vocab))
f.close()