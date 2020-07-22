# python David_1_11_4_approx_realworld.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/dog_contour.png"
# python David_1_11_4_approx_realworld.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/receipt.png
# python David_1_11_4_approx_realworld.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/a.jpg

    
    # import the necessary images
import cv2
import imutils   
import argparse

    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # load the image and convert it to grayscale,  and detect edges
image = cv2.imread(args["image"])
image = cv2.resize(image,(600,450), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)


    # show the original image and edged map
cv2.imshow("Original", image)
cv2.imshow("Edge Map", edged)
cv2.waitKey(0)

    # find contours in the image and sort them from largest to smallest,
    # keeping only the largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]

    # loop over the contours
for c in cnts:
	# approximate the contour and initialize the contour color
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)

	# show the difference in number of vertices between the original and approximated contours
	print("original: {}, approx: {}".format(len(c), len(approx)))

	# if the approximated contour has 4 vertices, then we have found our rectangle
	if len(approx) == 4:
		# draw the outline on the image
		cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
