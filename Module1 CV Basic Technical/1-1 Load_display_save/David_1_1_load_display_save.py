# !! Command mode(Terminal) Execute : python load_display_save.py --image grand_canyon.png 

# USAGE
# python David_1_1_load_display_save.py --image florida_trip.png

# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-i", "--image", required=True, help="Image name/ Type")

args = vars(ap.parse_args())

# load the image and show some basic information on it
image = cv2.imread(args["image"])
print("width: {w} pixels".format(w=image.shape[1]))
print("height: {h}  pixels".format(h=image.shape[0]))
print("channels: {c}".format(c=image.shape[2]))

# show the image and wait for a keypress
cv2.imshow("Image", image) 
cv2.waitKey(0) # cv2.waitKey :

'''
# cv2.waitKey :
    - cv2.waitKey 函數是用來等待與讀取使用者按下的按鍵，而其參數是等待時間（單位為毫秒），若設定為 0 就表示持續等待至使用者按下按鍵為止
    - https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
'''

# save the image -- OpenCV handles converting filetypes
# automatically
cv2.imwrite("./imwrite/man.jpg", image)




