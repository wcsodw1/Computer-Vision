# USAGE
# python David_1_10_2_1_auto_canny.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/RM.jpg"
# python David_1_10_2_1_auto_canny.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/a.jpg"


# python canny.py --image "../../../../CV-PyImageSearch Gurus Course/Dataset/data/coins01.png"

# 1.Preprocessing : 

    ## 1.1 import the necessary packages
import argparse
import imutils
import cv2

    ## 1.2 construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

    # 1.3 load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
image = cv2.resize(image,(450,300), interpolation = cv2.INTER_CUBIC)
cv2.imshow("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# 2.(!!) apply Canny edge detection
    # apply Canny edge detection using a wide threshold, tight threshold, and automatically determined threshold
    
    ## 2.1 Canny10(wide) = Thersohold門檻低(10, 200)
Canny10 = cv2.Canny(blurred, 10, 200)
cv2.imshow("Canny10", Canny10)

Canny50 = cv2.Canny(blurred,  50, 250)
Canny100 = cv2.Canny(blurred, 100, 250)
Canny150 = cv2.Canny(blurred, 150, 250)
Canny200 = cv2.Canny(blurred, 200, 250)
cv2.imshow("Canny50", Canny50)
cv2.imshow("Canny100", Canny100)
cv2.imshow("Canny150", Canny150)
cv2.imshow("Canny200", Canny200)

    ## 2.2 Canny225(Tight) = Thersohold門檻高(225, 250)
Canny225 = cv2.Canny(blurred, 225, 250)
cv2.imshow("Canny225", Canny225)

cv2.waitKey(0)
cv2.destroyAllWindows()



# 3.(!!!!) imutils.auto_canny(變數) : 自動邊緣檢測(Automatic Canny Edge Detection) 
# 效果 : 抓出邊緣 若已抓出邊緣的圖使用 會使得邊緣Double化
     
    #(!) 2.3 Auto_wide
auto_w = imutils.auto_canny(wide)
cv2.imshow("Wide", wide)
cv2.imshow("Auto_wide", auto_w)

    #(!) 2.4 Auto_biurred
auto_t = imutils.auto_canny(tight)
cv2.imshow("Tight", tight)
cv2.imshow("Auto_tight", auto_t)

cv2.waitKey(0)
cv2.destroyAllWindows()

    #(!) 2.5 Auto_biurred 
auto_b = imutils.auto_canny(blurred)
cv2.imshow("blurred", blurred)
cv2.imshow("Auto_blurred", auto_b)

cv2.waitKey(0)