# USAGE
# python David_1_9_3_adaptive_thresholding.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/license_plate.png"
# python David_1_9_3_adaptive_thresholding.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/a.jpg"

# 1.Preprocessing
    # 1.1 import the necessary packages
from skimage.filters import threshold_local
import argparse
import cv2

    # 1.2 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # 1.3 load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
image = cv2.resize(image,(400,600), interpolation = cv2.INTER_CUBIC)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
blurred7 = cv2.GaussianBlur(image, (7, 7), 0)

cv2.imshow("Image", image)
cv2.imshow("blurred5", blurred)
cv2.imshow("blurred7", blurred7)

# instead of manually specifying the threshold value, we can use adaptive
# thresholding to examine neighborhoods of pixels and adaptively threshold
# each neighborhood -- in this example, we'll calculate the mean value
# of the neighborhood area of 25 pixels and threshold based on that value;
# finally, our constant C is subtracted from the mean calculation (in this
# case 15)
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
cv2.imshow("OpenCV Mean Thresh", thresh)

# personally, I prefer the scikit-image adaptive thresholding, it just
# feels a lot more "Pythonic"
T = threshold_local(blurred, 29, offset=5, method="gaussian")
thresh = (blurred < T).astype("uint8") * 255
cv2.imshow("scikit-image Mean Thresh", thresh)
cv2.waitKey(0)
