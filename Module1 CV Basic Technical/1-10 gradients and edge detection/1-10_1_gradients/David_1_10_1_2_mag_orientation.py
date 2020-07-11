# python David_1_10_1_2_mag_orientation.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/girl.JPG"
# python David_1_10_1_2_mag_orientation.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/coins02.png"
# python David_1_10_1_2_mag_orientation.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/clonazepam_1mg.png"
# python David_1_10_1_2_mag_orientation.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/a.jpg"


# import the necessary packages
import numpy as np
import argparse
import cv2

# 1. Preproccesing : 

    # !! 1.1(Learn!!) construct the argument parser and parse the arguments    

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-l", "--lower-angle", type=float, default=175.0,help="Lower orientation angle")
ap.add_argument("-u", "--upper-angle", type=float, default=180.0,help="Upper orientation angle")
#ap.add_argument("-N","--NBA", type = int, default = 1, help ="ArgumentTest")

args = vars(ap.parse_args())

    ## 1.2 load the image, convert it to grayscale, and display the original image
image = cv2.imread(args["image"])
image = cv2.resize(image, (400,600), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
#cv2.waitKey(0)


# 2. compute gradients along the X and Y axis, respectively
gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
Combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
Combinedx = cv2.addWeighted(gX, 3, gY, 0.5, 0)
Combinedy = cv2.addWeighted(gX, 0.5, gY, 3, 0)

cv2.imshow("gX", gX)
cv2.imshow("gY", gY)
cv2.imshow("Combined", Combined)
cv2.imshow("Combinedx", Combinedx)
cv2.imshow("Combinedy", Combinedy)
cv2.waitKey(0)
cv2.destroyAllWindows()

    # 2.1 (POINT!!)compute the gradient magnitude and orientation, respectively
mag = np.sqrt((gX ** 2) + (gY ** 2))
magx = np.sqrt((gX ** 5) + (gY ** 2))
magy = np.sqrt((gX ** 2) + (gY ** 5))

cv2.imshow("mag", mag)
cv2.imshow("magx", magx)
cv2.imshow("magy", magy)
cv2.imwrite('../../data/imwrite/chp1_10_Grdient/magnitude/magx5y2.jpg', magx)
cv2.imwrite('../../data/imwrite/chp1_10_Grdient/magnitude/magy5x2.jpg', magy)
cv2.waitKey(0)
cv2.destroyAllWindows()

orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
orientation45 = np.arctan2(gY, gX) * (45 / np.pi) % 180
orientation20 = np.arctan2(gY, gX) * (20 / np.pi) % 180
#orientation20_90 = np.arctan2(gY, gX) * (20 / np.pi) % 30

cv2.imshow("orientation", orientation)
cv2.imshow("orientation45", orientation45)
cv2.imshow("orientation20", orientation20)
cv2.imshow("orientation20_90", orientation20_90)


cv2.waitKey(0)
cv2.destroyAllWindows()

    # 2.2 find all pixels that are within the upper and low angle boundaries
idxs = np.where(orientation >= args["lower_angle"], orientation, -1)
idxs = np.where(orientation <= args["upper_angle"], idxs, -1)
mask = np.zeros(gray.shape, dtype="uint8")
mask[idxs > -1] = 255

cv2.imshow("Mask", mask)
cv2.waitKey(0)


idxs = np.where(orientation45 >= args["lower_angle"], orientation45, -1)
idxs = np.where(orientation45 <= args["upper_angle"], idxs, -1)
mask45 = np.zeros(gray.shape, dtype="uint8")
mask45[idxs > -1] = 255
cv2.imwrite('../../data/imwrite/chp1_10_Grdient/magnitude/mask45.jpg', mask45)
cv2.imshow("Mask45", mask45)
cv2.waitKey(0)

idxs = np.where(orientation20 >= args["lower_angle"], orientation20, -1)
idxs = np.where(orientation20 <= args["upper_angle"], idxs, -1)
mask20 = np.zeros(gray.shape, dtype="uint8")
mask20[idxs > -1] = 255

cv2.imshow("Mask20", mask20)
cv2.imwrite('../../data/imwrite/chp1_10_Grdient/magnitude/mask20.jpg', mask20)
cv2.waitKey(0)