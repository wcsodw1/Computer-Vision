
# python David_1_7_1_bilateral.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/family.jpg"


# 1.Preprocessing
    # import the necessary packages
import argparse
import cv2

    ## construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    ## load the image, display it, and construct the list of bilateral
# filtering parameters that we are going to explore
image = cv2.imread(args["image"])
image = cv2.resize(image,(450,300), interpolation = cv2.INTER_CUBIC)
cv2.imshow("Original", image)



# 2.Define the Filter 
    ## 2.1 Suppose we have N(4) filter 
filters = [(10,30,20),(20,60,40),(30,90,60),(50,150,100)]

for (diameter, sigmaColor, sigmaSpace) in filters:  # for (list中某項裡面多個參數)
    blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)  
    title = "Blurred A={}, B={}, C={}".format(diameter, sigmaColor, sigmaSpace)
    cv2.imshow(title, blurred)
    cv2.waitKey(0)

# params = [ (11, 41, 21), (11, 61, 39), (11, 100, 60),]

# # loop over the diameter, sigma color, and sigma space
# for (diameter, sigmaColor, sigmaSpace) in params:
# 	# apply bilateral filtering and display the image
# 	blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
# 	title = "Blurred d={}, sc={}, ss={}".format(diameter, sigmaColor, sigmaSpace)
# 	cv2.imshow(title, blurred)
# 	cv2.waitKey(0)



