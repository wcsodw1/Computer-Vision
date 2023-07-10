
# python David_1_6_morphological.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/AUS.JPG"
# python David_1_6_morphological.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/a.JPG"


# 1. Preprocessing
    # import the necessary packages
import argparse
import cv2

    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.resize(image,(400,450), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.waitKey(0)

#2. Multiple Morphological operations (形態學轉換)

#    # 2.1 gradient　形態學梯度
# kernelSizes = [(3, 3), (5, 5), (7, 7)]

# cv2.imshow("Original", image)
# for kernelSize in kernelSizes:
# 	gradient = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
# 	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, gradient)
# 	cv2.imshow("gradient: ({},{})".format(kernelSize[0], kernelSize[1]), gradient)
#     #cv2.imwrite("../../data/imwrite/chp1_6_morphological/Gradient({},{}).jpg".format(kernelSize[0], kernelSize[1]), gradient)
# 	cv2.waitKey(0)
# cv2.destroyAllWindows()
    


#     # 2.2 tophat　禮帽 
# cv2.imshow("Original", image)
# for kernelSize in kernelSizes:
#     tophat = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
#     tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat)
#     cv2.imshow("tophat: ({},{})".format(kernelSize[0], kernelSize[1]), tophat)
#     cv2.imwrite("../../data/imwrite/chp1_6_morphological/Tophat({},{}).jpg".format(kernelSize[0], kernelSize[1]),tophat)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


 
#     # 2.3 blackhat　黑帽    
# cv2.imshow("Original", image)
# for kernelSize in kernelSizes:
#     BLACKHAT = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
#     BLACKHAT = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, BLACKHAT)
#     cv2.imshow("BlackHat:({},{})".format(kernelSize[0], kernelSize[1]), BLACKHAT)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()



#     # 2.4 cv2.add 將(BLACKHAT, tophat, gradient)相加　cv2.add(開運算, 禮帽)
# cv2.imshow("Original", image)
# for kernelSize in kernelSizes:
#     BLACKHAT_and_tophat  = cv2.add(tophat, BLACKHAT)
#     BLACKHAT_and_tophat_gradient = cv2.add(gradient, BLACKHAT_and_tophat)
#     cv2.imshow("Blackhat_and_tophat_gradient:({},{})".format(kernelSize[0], kernelSize[1]), BLACKHAT_and_tophat_gradient)
#     cv2.waitKey(0)
#     cv2.imwrite("../../data/imwrite/chp1_6_morphological/Blackhat_and_tophat_gradient({},{}).jpg".format(kernelSize[0], kernelSize[1]), BLACKHAT_and_tophat_gradient)


# # apply a series of erosions
# for i in range(0, 5):
# 	eroded = cv2.erode(gray.copy(), None, iterations=i + 1)
# 	cv2.imshow("Eroded {} times".format(i + 1), eroded)
# 	cv2.waitKey(0)

# # close all windows to cleanup the screen
# cv2.destroyAllWindows()


# cv2.imshow("Original", image)

# apply a series of dilations
for i in range(0, 5):
	dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
	cv2.imshow("Dilated {} times".format(i + 1), dilated)
	cv2.waitKey(0)
    
    
# # close all windows to cleanup the screen and initialize the list
# of kernels sizes that will be applied to the image
cv2.destroyAllWindows()
cv2.imshow("Original", image)
kkS = [(3, 3), (5, 5), (7, 7),(10,10),(12,12),(15,15)]
# loop over the kernels and apply an "opening" operation to the image
for kernelSize in kkS:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(0)
    
    
    
# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over the kernels and apply a "closing" operation to the image
kS = [(3, 3), (5, 5), (7, 7),(10,10)]

for kernelSize in kS:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("Closing: ({}, {})".format(kernelSize[0], kernelSize[1]), closing)
	cv2.waitKey(0)

# close all windows to cleanup the screen
cv2.destroyAllWindows()
