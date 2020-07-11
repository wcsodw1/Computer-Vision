# 大津演算法 : https://zh.wikipedia.org/wiki/%E5%A4%A7%E6%B4%A5%E7%AE%97%E6%B3%95
# python David_1_9_1_otsu_thresholding.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/coins01.png"
# python David_1_9_1_otsu_thresholding.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/a.jpg"


# 1.Preprocessing
    # 1.1 import the necessary packages
import argparse
import cv2

    # 1.2 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # 1.3 load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
image = cv2.resize(image, (400,500), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow("Image", image)
cv2.imshow("gray", gray)
cv2.imshow("blurred", blurred)

#(Point!!!!) 2. apply Otsu's automatic thresholding 
    # 2.1 Otsu's method automatically ,determines the best threshold value `T` for us
    
(T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("blurred_Threshold", threshInv)
print("Otsu's thresholding value: {}".format(T))

    # 2.2 遮罩覆蓋上去 finally, we can visualize only the masked regions in the image
cv2.imshow("Mask_Blurred on Origin_Image", cv2.bitwise_and(image, image, mask=threshInv))
cv2.waitKey(0)


# 3. 網路上找到的其餘應用
from matplotlib import pyplot as plt

img = cv2.imread(args["image"])
img = cv2.resize(image, (600,800), interpolation = cv2.INTER_CUBIC)

ret , thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret , thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret , thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret , thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret , thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['original image','Binary','binary-inv','trunc','tozero','tozero-inv']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
plt.close() 


