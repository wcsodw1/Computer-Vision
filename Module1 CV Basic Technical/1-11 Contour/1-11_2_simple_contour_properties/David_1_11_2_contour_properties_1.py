
# python David_1_11_2_contour_properties_1.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/more_shapes.png"
# reference : https://chtseng.wordpress.com/2016/12/05/opencv-contour%E8%BC%AA%E5%BB%93/

# 1.Preprocessing : 
    # 1.1import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

    # 1.2 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
    # 1.3 load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.resize(image, (280,330), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #(!) 1.4 find external contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("cnts : " , cnts)

#cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("cnts : " , cnts)
clone = image.copy()


# 2.draw the center of the contour on the image by loop

for i in cnts:
	# compute the moments of the contour which can be used to compute the centroid or "center of mass" of the region

"""一）標示中心點：

要取得Contour中心點，可使用OpenCV的moments（矩）函式，這是一個關於矩的計算函式。矩，又稱動差，英文為moment，"""

	M = cv2.moments(i) 
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the center of the contour on the image
	cv2.circle(clone, (cX, cY), 10, (0, 255, 0), -1)
print(cX) # int , 188
print(cY) # int , 155

cv2.imshow("Centroids", clone)
cv2.waitKey(0)
clone = image.copy()


# 3. compute the area and the perimeter of the contour by loop over the contours again

for (i, c) in enumerate(cnts):
	# compute the area and the perimeter of the contour
	area = cv2.contourArea(c)
	perimeter = cv2.arcLength(c, True)
	print("Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(i + 1, area, perimeter))

	# draw the contour on the image
	cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)

	# compute the center of the contour and draw the contour number
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	cv2.putText(clone, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (255, 255, 255), 4)

cv2.imshow("Contours", clone)
cv2.waitKey(0)


# 4. cv2.boundingRect : fit a bounding box to the contour
    # clone the original image
clone = image.copy()

# loop over the contours
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c) # fit a bounding box to the contour
	cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Bounding Boxes", clone)
cv2.waitKey(0)

# 5. cv2.minAreaRect(c) : fit a rotated bounding box to the contour and draw a rotated bounding box
clone = image.copy()

# loop over the contours
for c in cnts:
	box = cv2.minAreaRect(c) # fit a rotated bounding box to the contour and draw a rotated bounding box
	box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
	cv2.drawContours(clone, [box], -1, (0, 255, 0), 2)

cv2.imshow("Rotated Bounding Boxes", clone)
cv2.waitKey(0)

# 6. fit a minimum enclosing circle to the contour
clone = image.copy()

# loop over the contours
for c in cnts:
	# fit a minimum enclosing circle to the contour
	((x, y), radius) = cv2.minEnclosingCircle(c)
	cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)

cv2.imshow("Min-Enclosing Circles", clone)
cv2.waitKey(0)


# 7. to fit an ellipse, our contour must have at least 5 points

clone = image.copy()
for c in cnts: 
	if len(c) >= 5: #to fit an ellipse, our contour must have at least 5 points
		# fit an ellipse to the contour
		ellipse = cv2.fitEllipse(c)
		cv2.ellipse(clone, ellipse, (0, 255, 0), 2)
        
cv2.imshow("Ellipses(橢圓)", clone)
cv2.waitKey(0)