# python David_1_7_2_blurring.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/AUS1.jpg"
# python David_1_7_2_blurring.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/a.jpg"


# 1.Preprocessing

    # import the necessary packages
import argparse
import cv2

    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # load the image, display it, and initialize the list of kernel sizes
image = cv2.imread(args["image"])
image = cv2.resize(image,(300,300), interpolation = cv2.INTER_CUBIC)
cv2.imshow("Original", image)


# 2.Define the Filter 
    # Suppose we have N(4) filter()
    
kernelSizes = [(3, 3), (9, 9), (15, 15)]

    
    ## 2.1 loop over the kernel sizes and apply an "average" blur to the image
for (kX, kY) in kernelSizes:
	blurred = cv2.blur(image, (kX, kY))
	cv2.imshow("Average ({},{})".format(kX, kY), blurred)
    #cv2.imwrite("Average ({},{}).jpg".format(kX, kY), blurred)
	cv2.waitKey(0)
cv2.destroyAllWindows()

    ## 2.2 loop over the kernel sizes and apply a "Gaussian" blur to the image
cv2.imshow("Original", image)
for (kX, kY) in kernelSizes:
	blurred = cv2.GaussianBlur(image, (kX, kY), 0)
	cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)
 
	cv2.waitKey(0)
cv2.destroyAllWindows()

    # 2.3 loop over the kernel sizes and apply a "Median" blur to the image
cv2.imshow("Original", image)
for k in (3, 9, 15):
	blurred = cv2.medianBlur(image, k)
	cv2.imshow("Median {}".format(k), blurred)
	cv2.waitKey(0)
    
    