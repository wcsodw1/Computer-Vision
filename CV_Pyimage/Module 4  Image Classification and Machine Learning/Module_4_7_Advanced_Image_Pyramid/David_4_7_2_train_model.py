# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:50:26 2020

@author: user
"""
# python David_4_7_2_train_model.py --dataset output/data/training --features-db output/training_features.hdf5 --pbow-db output/training_pbow_.hdf5 --model output/model_l4.cpickle
# python David_4_7_2_train_model.py 

    # import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import sklearn
import numpy as np
import argparse
import pickle
import h5py
import cv2
import sys
        # (!) handle sklearn versions less than 0.18
if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.grid_search import GridSearchCV

    # (!!)otherwise, sklearn.grid_search is deprecated
    # and we'll import GridSearchCV from sklearn.model_selection
else:
	from sklearn.model_selection import GridSearchCV
    
    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the original images")
ap.add_argument("-f", "--features-db", required=True,
	help="Path the features database")
ap.add_argument("-p", "--pbow-db", required=True,
	help="Path to where the pyramid of bag-of-visual-words database")
ap.add_argument("-m", "--model", required=True,
	help="Path to the output classifier")

sys.argv[1:] = '-d output/data/training -f output/training_features.hdf5 -p output/training_pbow_L4.hdf5 -m output/model_L4.cpickle'.split()
args = vars(ap.parse_args())

# open the features and bag-of-visual-words databases
featuresDB = h5py.File(args["features_db"])
bovwDB = h5py.File(args["pbow_db"])

# grab the training and testing data from the dataset using the first 300
# images as training and the remaining images for testing
print("[INFO] loading data...")
(trainData, trainLabels) = (bovwDB["bovw"][:300], featuresDB["image_ids"][:300])
(testData, testLabels) = (bovwDB["bovw"][300:], featuresDB["image_ids"][300:])

# prepare the labels by removing the filename from the image ID, leaving
# us with just the class name
trainLabels = [l.split("\\")[1] for l in trainLabels]
print("trainLabels : ", trainLabels)
print("")
testLabels = [l.split("\\")[1] for l in testLabels]
print("testLabels:",testLabels)
# define the grid of parameters to explore, then start the grid search where
# we evaluate a Linear SVM for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LinearSVC(random_state=42), params, cv=3)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))


# show a classification report
print("[INFO] evaluating...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# loop over a sample of the testing data
for i in np.random.choice(np.arange(300, 375), size=(20,), replace=False):
	# randomly grab a testing image, load it, and classify it
	(label, filename) = featuresDB["image_ids"][i].split(":")
	label ='output\\' + label
	print(label)
	image = cv2.imread("{}/{}".format(label, filename))
	print(image)
	prediction = model.predict(bovwDB["bovw"][i].reshape(1, -1))[0]
	print(prediction)
	# show the prediction
	print("[PREDICTION] {}:{}".format(filename, prediction))
	cv2.putText(image, prediction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
	image = cv2.resize(image, (450,600), interpolation = cv2.INTER_CUBIC)
	cv2.imshow("Image", image)
	cv2.waitKey(2000)	   
	cv2.destroyAllWindows()


# close the databases
featuresDB.close()
bovwDB.close()

# dump the classifier to file
print("[INFO] dumping classifier to file...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()