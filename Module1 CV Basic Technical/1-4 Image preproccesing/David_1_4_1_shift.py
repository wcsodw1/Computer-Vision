# python David_1_4_1_shift.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/aus_beach.JPG"


import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help = "Path to image")
args = vars(ap.parse_args())

# Load image
img = cv2.imread(args["image"])
img = cv2.resize(img, (450,330), interpolation = cv2.INTER_CUBIC)
cv2.imshow("Aus_Beech", img)


# Try-myself
M = np.float32([[1, 0, -100], [0, 1, -150]])
shifted = imutils.translate(img, 0, 100)
cv2.imshow("Test Shifted", shifted)


M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Shifted Down and Right", shifted)

shifted = imutils.translate(img, 0, -100)
cv2.imshow("Shifted up", shifted)
cv2.waitKey(0)