# python David_1_11_1_1_finding_drawing_contours.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/basic_shapes.png"

    # 1.import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # load the image and convert it to grayscale
image = cv2.imread(args["image"])
#image = cv2.resize(image,(400,600), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original image
cv2.imshow("Original", image)

# 2.find all contours(輪廓)

    #(!!) 2.1 find all contours in the image and 
cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(type(cnts))  # tuple
cnts = imutils.grab_contours(cnts)
print(type(cnts)) # list


    #(!!!) 2.2 draw ALL contours on the image 
    
clone = image.copy() #為了不覆蓋掉image, copy一個clone去做事(so clone就是image)
print(clone.shape)
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print("Found {} contours".format(len(cnts)))
cv2.imshow("All Contours", clone)
cv2.waitKey(0)

#--------------------------------------------------------

# 3. loop the contour region one-by-one(一個一個將contour描繪出來)
    
    ## 3.1 loop over the contours individually and draw each of them(一個一個將contour描繪出來)
clone = image.copy()
cv2.destroyAllWindows()
    #(!!!) learn how to make loop  
for (i, j) in enumerate(cnts):
    print("Contour #{}".format(i + 1))
    cv2.drawContours(clone, [j], -1,(0,255,0),2)
    cv2.imshow("Single Contour", clone)
    cv2.waitKey(0)

    # 3.2 find contours in the image, but this time keep only the EXTERNAL(外部) contours in the image
clone = image.copy()
cv2.destroyAllWindows()
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print("Found {} EXTERNAL contours".format(len(cnts)))
cv2.imshow("All Contours", clone)
cv2.waitKey(0)

    # 3.3 oop over the contours individually(用for一個一個visualize 1.mask與2.mask遮罩融合原影像(image + mask)的圖顯示出來)
    # construct a mask by drawing only the current contour

clone = image.copy()
cv2.destroyAllWindows()
cv2.imshow("Image", image)

for i in cnts:
	mask = np.zeros(gray.shape, dtype = "uint8")
	cv2.drawContours(mask, [i], -1, 255, -1)
	cv2.imshow("Mask", mask)
	cv2.imshow("Image + Mask", cv2.bitwise_and(image, image, mask=mask))
	cv2.waitKey(0)
