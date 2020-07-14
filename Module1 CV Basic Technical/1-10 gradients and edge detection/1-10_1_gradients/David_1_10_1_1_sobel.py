# USAGE
# python David_1_10_1_1_sobel.py --image "../../../../CV_PyImageSearch/Dataset/data/jared.JPG"
# python David_1_10_1_1_sobel.py --image "../../../../CV_PyImageSearch/Dataset/data/opera.JPG"
# python David_1_10_1_1_sobel.py --image "../../../../CV_PyImageSearch/Dataset/data/a.jpg"

# import the necessary packages
import argparse
import cv2

# 1. Preproccesing : 

    ## 1.1 construct the argument parser and parse the arguments    
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

    ## 1.2 load the image, convert it to grayscale, and display the original image
image = cv2.imread(args["image"])
image = cv2.resize(image, (500,400), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)



# 2. compute gradients along the X and Y axis, respectively (use gray instead of image)
gX = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx = 1, dy = 0)
gY = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx = 0, dy = 1)
cv2.imshow("gX", gX)
cv2.imshow("gY", gY)
#(TRY)combine the sobel X and Y representations into a single image
sobelCombined = cv2.addWeighted(gX, 0.1, gY, 0.1, 0)
cv2.imshow("Combined", sobelCombined)

cv2.imwrite('../../data/imwrite/chp1_10_Grdient/opera_cv2_Sobel.jpg', gX)
cv2.waitKey(0)
cv2.destroyAllWindows()

    ## 2.1 實現圖像增強 (cv2.convertScaleAbs)
    # the `gX` and `gY` images are now of the floating point data type,
    # so we need to take care to convert them back a to unsigned 8-bit
    # integer representation so other OpenCV functions can utilize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)
cv2.imshow("ConvertgX", gX)
cv2.imshow("ConvertgY", gY)
cv2.imwrite('../../data/imwrite/chp1_10_Grdient/opera_Convet.jpg', gX)
cv2.waitKey(0)
cv2.destroyAllWindows()

    ## 2.2 圖片設置透明度疊加 cv2.addWeighted
    ## 權重越大 透明度越低
    ## (Try Different way) combine the sobel X and Y representations into a single image
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
sobelCombinedX = cv2.addWeighted(gX, 3, gY, 0.1, 0)
sobelCombinedY = cv2.addWeighted(gX, 0.1, gY, 3, 0)

    # show our output images
cv2.imshow("Sobel X", gX)
cv2.imshow("Sobel Y", gY)

cv2.imshow("Sobel Combined", sobelCombined)
cv2.imshow("Sobel CombinedX", sobelCombinedX)
cv2.imshow("Sobel CombinedY", sobelCombinedY)
cv2.imwrite('../../data/imwrite/chp1_10_Grdient/Opera_addWeighted.jpg', sobelCombinedX)
cv2.waitKey(0)