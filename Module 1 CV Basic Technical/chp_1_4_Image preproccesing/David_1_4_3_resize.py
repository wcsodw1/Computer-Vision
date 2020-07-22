# python David_1_4_3_resize.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/RM.jpg"

import argparse
import imutils
import cv2


# 1.Preprocessing

    # 1.1 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # 1.2 load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)


# 2.Multiple Resize function below : 

    # 2.1 calculate the ratio of the new image : 
    # we need to keep in mind aspect ratio so the image does not look skewed
    # or distorted -- therefore, we calculate the ratio of the new image to
    # the old image. Let's make our new image have a width of 150 pixels
r = 200.0 / image.shape[1]
dim = (300, int(image.shape[0] * r))

    # perform the actual resizing of the image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)

    # 2.2 Adjust the height and Width by ratio : 
    # what if we wanted to adjust the height of the image? -- we can apply
    # the same concept, again keeping in mind the aspect ratio, but instead
    # calculating the ratio based on height -- let's make the height of the
    # resized image 50 pixels
r = 50.0 / image.shape[0]
dim = (int(image.shape[1] * r), 50)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Height)", resized)

    # 2.3 resize width/ helght directly by value(直接數值轉換)
    # of course, calculating the ratio each and every time we want to resize
    # an image is a real pain -- let's create a  function where we can specify
    # our target width or height, and have it take care of the rest for us.
resized = imutils.resize(image, height=300)
cv2.imshow("Resized height via Function", resized)
    
resized = imutils.resize(image, width=100)
cv2.imshow("Resized width via Function", resized)
cv2.waitKey(0)


