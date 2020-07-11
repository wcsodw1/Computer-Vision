# python David_1_4_2_rotate.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/Ferris_Wheel.JPG"

import numpy as np 
import argparse
import cv2
import imutils


# 1.Preprocessing

    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # load the image and show it
image = cv2.imread(args["image"])
image = cv2.resize(image,(450,300), interpolation = cv2.INTER_CUBIC )
cv2.imshow("Original", image)


# 2.grab the dimensions of the image and calculate the center of the image
(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)


    # 2.1 rotate our image by 45 degrees
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)

    # 2.2 rotate our image by -90 degrees
M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -90 Degrees", rotated)

    # 2.3 rotate our image around an arbitrary point rather than the center
M = cv2.getRotationMatrix2D((cX - 50, cY - 50), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by Offset & 45 Degrees", rotated)

# finally, let's use our helper function in imutils to rotate the image by
    # 2.4 180 degrees (flipping it upside down)
rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated by 180 Degrees", rotated)

cv2.waitKey(0)
