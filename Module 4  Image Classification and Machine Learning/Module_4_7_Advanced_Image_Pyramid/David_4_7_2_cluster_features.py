# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:55:29 2020

@author: user
"""
# python David_4_7_2_cluster_features.py --features-db output/training_features.hdf5 --codebook output/vocab.cpickle --clusters 512 --percentage 0.25
# python David_4_7_2_cluster_features.py

    # import the necessary packages
from __future__ import print_function
from pyimagesearch.ir import Vocabulary
import argparse
import pickle
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,
	help="Path to where the features database will be stored")
ap.add_argument("-c", "--codebook", required=True,
	help="Path to the store output codebook")
ap.add_argument("-k", "--clusters", type=int, default=512,
	help="# of clusters to generate")
ap.add_argument("-p", "--percentage", type=float, default=0.25,
	help="Percentage of total features to use when clustering")

#import sys
sys.argv[1:] = '-f output/training_features.hdf5 -c output/vocab.cpickle -k 512 -p 0.25'.split()
args = vars(ap.parse_args())

# create the visual words vocabulary
voc = Vocabulary(args["features_db"])
print(voc)
vocab = voc.fit(args["clusters"], args["percentage"])
print(vocab)

# dump the clusters to file
print("[INFO] storing cluster centers...")
f = open(args["codebook"], "wb")
f.write(pickle.dumps(vocab))
f.close()