
# python David_1_4_6_arithmetic.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/jared.JPG"
# python David_1_4_6_arithmetic.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/grand_canyon.png"


# import the necessary packages
import numpy as np
import argparse
import cv2

# 1.Preprocessing
    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # load the image and show it
image = cv2.imread(args["image"])
image = cv2.resize(image,(200,300),interpolation = cv2.INTER_CUBIC)
cv2.imshow("Original", image)


# 2.加減Channel於影像上 : 

    # 2.1 加 chanel
M = np.ones(image.shape, dtype = "uint8") * 80
added = cv2.add(image, M)
cv2.imshow("Added80", added)

    # 2.1 加 chanel
M = np.ones(image.shape, dtype = "uint8") * 30
added = cv2.add(image, M)
cv2.imshow("Added30", added)

    # 2.2 減 chanel
M = np.ones(image.shape, dtype = "uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted50", subtracted)

M = np.ones(image.shape, dtype = "uint8") * 100
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted100", subtracted)
cv2.waitKey(0)



