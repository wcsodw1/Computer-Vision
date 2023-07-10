# python David_1_13_connected_components_labeling.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/tictactoe.png"
# python David_1_13_connected_components_labeling.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/license_plate.png"
# https://blog.csdn.net/sinat_21258931/article/details/61418 

# 1. Preprocessing : 

    # 1.1 import the necessary packages
from skimage import measure
import numpy as np
import cv2
import argparse


    # 1.2 load the license plate image from disk
    # extract the Value component from the HSV color space 
#plate = cv2.imread("../../data/license_plate.png") # 這範例也可行!

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

plate = cv2.imread(args["image"])
print(plate.shape) # (480, 440, 3)

V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2] # 取其中一維的值
print("V :" , V) #
print("Type V : ", type(V))  # <class 'numpy.ndarray'>
print(V.shape) # (480, 440)
cv2.imshow("Original-License Plate", plate)

    # 1.3  用threshold幫助我們方便抓取label數
    # 領域內均值 : ADAPTIVE_THRESH ,apply adaptive thresholding to reveal the characters on the license plate
# thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
# print("Type thresh :" , type(thresh))
# cv2.imshow("Thresh", thresh)

thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 17, 3)
print("Type thresh :" , type(thresh))
cv2.imshow("Thresh", thresh)

# 2. measure.label / 
    # 1.4(!!!) measure.label  : 對二值圖像進行聯通區域標記 - 
    # measure.label : https://blog.csdn.net/pursuit_zhangyu/article/details/94209489
   
    # perform connected components analysis on the thresholded images and initialize the
    # mask to hold only the "large" components we are interested in
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
    # 2.1 找出 Label總數值
print("[INFO] found {} blobs".format(len(np.unique(labels))))

    # 使用迴圈跑出以下.... loop over the unique components
for (i, label) in enumerate(np.unique(labels)):
    
    # 2.2 背景為label的0, if this is the background label, ignore it
	if label == 0:
		print("[INFO] label: 0 (background)")
		continue

    # 2.3 依序列出 Label數
	# otherwise, construct the label mask to display only connected components for the current label
	print("[INFO] label: {} (foreground)".format(i))
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
   
	# 2.4  較大的部件我們將他add mask 並一同顯示出來
    # if the number of pixels in the component is sufficiently large, add it to our mask of "large" blobs
	if numPixels > 300 and numPixels < 1500:
		mask = cv2.add(mask, labelMask)

    # show the label mask
	cv2.imshow("Label", labelMask)
	cv2.waitKey(0)

# show the large components in the image
cv2.imshow("Large Blobs", mask)
cv2.waitKey(0)
