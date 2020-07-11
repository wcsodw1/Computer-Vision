
# python David_1_2_ImageCutting.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/a.jpg"
# python (檔名.py) -- (路徑(影像名稱.jpg.png) or....)


# 1.import the necessary packages
import argparse
import cv2

# 2.argparse : 可使程式能夠直接在Terminal上面輸入檔案名稱(EX:(image).JPG) 
    # construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

    # load the image, grab its dimensions, and show it
image = cv2.imread(args["image"])
image = cv2.resize(image, (600, 500), interpolation=cv2.INTER_CUBIC)

(h, w) = image.shape[:2]  # Grab 3 Dimension


    # Visualize
cv2.imshow("Origin_Image", image)
print(image.shape)

    ## Print image Array : (來瞧瞧image裡面都是甚麼 其實就是3維的矩陣(x,y,z)) 
    ## images are just NumPy arrays. The top-left pixel can be found at (0, 0)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}".format(r=r, g=g, b=b))

    ## now, let's change the value of the pixel at (0, 0) and make it red
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}".format(r=r, g=g, b=b))


# compute the center of the image, which is simply the width and height
# divided by two

#-------------------------------------------------


# 3. Segmentation image to 3*3 (Slice to 9 Piece, 切成9塊)
(cLX, cTY) = (w // 3, h // 3)
(cRX, cBY) = (w*2 // 3, h*2 // 3)

    # Top 
tl = image[0:cTY, 0:cLX]
tm = image[0:cTY, cLX:cRX]
tr = image[0:cTY, cRX:w]

cv2.imshow("Top-Left Corner", tl)
cv2.imwrite("./imwrite/top-left.jpg", tl)

cv2.imshow("Top-Middle", tm)
cv2.imwrite("./imwrite/top-Mid.jpg", tm)

cv2.imshow("Top-Right Corner", tr)
cv2.imwrite("./imwrite/top-right.jpg", tr)


    # Middle 
ml = image[cTY:cBY, 0:cLX]
mm = image[cTY:cBY, cLX:cRX]
mr = image[cTY:cBY, cRX:w]

cv2.imshow("Mid-Left Corner", ml)
cv2.imwrite("./imwrite/Mid-left.jpg", ml)
cv2.imshow("Mid-Middle", mm)
cv2.imwrite("./imwrite/Mid-Mid.jpg", mm)
cv2.imshow("Mid-Right Corner", mr)
cv2.imwrite("./imwrite/Mid-Right.jpg", mr)


    # Bottom
bl = image[cBY:h, 0:cLX]
bm = image[cBY:h, cLX:cRX]
br = image[cBY:h, cRX:w]

cv2.imshow("Bottom-Left Corner", bl)
cv2.imwrite("./imwrite/bottom-left.jpg", bl)

cv2.imshow("Bottom-Middle", bm)
cv2.imwrite("./imwrite/bottom-Mid.jpg", bm)

cv2.imshow("Bottom-Right Corner", br)
cv2.imwrite("./imwrite/bottom-Right.jpg", br)




# 4. now let's make the top-left corner of the original image green
image[0:cLX, 0:cBY] = (0, 255, 0)

# Show our updated image
cv2.imshow("Mask", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./imwrite/0_Mask.jpg", image)


#Try to resize : 
# import numpy as np
# image = np.resize(image, new_shape=[256, 480, 3])
# image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_CUBIC)
# image = cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)  #縮放比例因子：fx=2,fy=2
# image = cv2.resize(image,None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC)  #比例因子：fx=0.5,fy=0.5
# height, width = image.shape[:2]
# image = cv2.resize(image, (int(0.5 * width), int(0.5 * height)), interpolation=cv2.INTER_CUBIC)  # dsize=（0.5*width,0.5*height）

# Try loop to save the image
# list_ = [tl,tm,tr]
# for i in range(len(list_)) : 
#     x = list_[i]
#     xy = str(list_[(i)])
#     cv2.imwrite( 'a' + xy,tr)
#     cv2.imwrite("segment[i].jpg",i )
