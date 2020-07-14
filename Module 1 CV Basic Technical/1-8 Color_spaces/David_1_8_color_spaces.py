# python David_1_8_color_spaces.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/a.JPG"

import argparse
import cv2

# 1. Preproccesing : 

    ## 1.1 construct the argument parser and parse the arguments    
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

    ## 1.2 load the original image and show it (RGB)
image = cv2.imread(args["image"])
image = cv2.resize(image,(300,450), interpolation = cv2.INTER_CUBIC)
cv2.imshow("RGB", image)

    # (for with Zip : to manage list in for loop !) loop over each of the individual channels and display them
for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
	cv2.imshow(name, chan)

    
print(cv2.split(image)) 
print(type(cv2.split(image))) # list
cv2.waitKey(0)
cv2.destroyAllWindows()


# 2.(point!!! HSV) convert the image to the HSV color space and show it
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

    # loop over each of the invidiaul channels and display them
for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
	cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 3.(point!!! LAB -> cv2.COLOR_BGR2LAB) convert the image to the L*a*b* color space and show it
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)

for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab)):
	cv2.imshow(name, chan)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 4. show the original and grayscale versions of the image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Original", image)
# cv2.imshow("Grayscale", gray)
# cv2.waitKey(0)

