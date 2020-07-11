
# python David_1_4_5_crop.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/MEL.JPG"

import cv2
import argparse

    # 1 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # 1.1 load the image and show it
image = cv2.imread(args["image"])
image = cv2.resize(image,(450,600), interpolation=cv2.INTER_CUBIC)
cv2.imshow("man", image)

    # 1.2 body
body = image[130:600,120:340]
cv2.imshow("body",body)
cv2.imwrite("../../../data/imwrite/chp1_4_5/David_MEL_body.jpg", body)


    # 1.3 face
face = image[130:220,120:200]
cv2.imshow("face",face)
cv2.waitKey(0)
cv2.imwrite("../../../data/imwrite/chp1_4_5/David_MEL_face.jpg", face)
