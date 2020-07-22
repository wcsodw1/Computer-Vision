# python David_1_4_4_flipping.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/girl.JPG"


import argparse
import cv2

# 1.Preprocessing

    # 1.1 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # 1.2 load the image and show it
image = cv2.imread(args["image"])
image = cv2.resize(image, (400,600), interpolation=cv2.INTER_CUBIC)
cv2.imshow("Original", image)


# 2.flipping

    # 2.1 flip the image horizontally
flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)

    # 2.2 flip the image vertically
flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)

    # 2.3 flip the image along both axes
flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally & Vertically", flipped)
cv2.waitKey(0)
