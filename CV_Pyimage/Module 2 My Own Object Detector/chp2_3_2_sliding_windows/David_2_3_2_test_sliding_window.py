# USAGE
# python David_2_3_2_test_sliding_window.py --image florida_trip.png --width 64 --height 64


# python David_2_3_2_test_sliding_window.py --image "../../../CV_PyImageSearch/Dataset/data/AUS1.JPG" --Width 100 --Height 100 -S 1.1
# python David_2_3_2_test_sliding_window.py --image "../../../CV_PyImageSearch/Dataset/data/basketball.jpg" --Width 100 --Height 100 -S 1.1 -st 200
# python David_2_3_2_test_sliding_window.py --image "../../../CV_PyImageSearch/Dataset/data/family.jpg" --Width 200 --Height 50 -S 2 -st 50


# 1.Preprocessing : 

    # 1.1 import the necessary packages
from pyimagesearch.object_detection.helpers import sliding_window
from pyimagesearch.object_detection.helpers import pyramid
import argparse
import time
import cv2

    # 1.2 (!!!) 增加 width/height/scale 等三種參數, 
    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-W", "--Width", required = True, type = int, help = "Width of Sliding Window")
ap.add_argument("-H", "--Height",required = True, type = int, help = "Height of sliding Window" )
ap.add_argument("-S", "--Scale" , type = float, default = 1.2, help = "Scale factor Size")  # Scale : 縮放倍率 Zoom in Scale
ap.add_argument("-st", "--stepsize" , type = int, default = 64, help = "Step Size")  # stepsize : 步伐

args = vars(ap.parse_args())

    # 1.3 load the input image and unpack the command line arguments
image = cv2.imread(args["image"])
image = cv2.resize(image,(800,650), interpolation = cv2.INTER_CUBIC)
(winW, winH) = (args["Width"], args["Height"])




# 2. 用迴圈跑整個image ,並縮小image繼續跑 直至收斂 : 

    # 2.1 用迴圈跑整個image,完成後每次經過scale縮小後繼續直到收斂 
    # loop over the image pyramid
for layer in pyramid(image, scale = args["Scale"]):
    
    # 2.2 用迴圈跑每個image 並設定每次的移動距離(stepSize) , 及 BoundingBoxWindow的大小
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(layer, stepSize = args["stepsize"], windowSize=(winW, winH)): # sliding_window : Library
        
        # 2.3 BoundingBoxWindow 若無走到預期位置 則跳出進入下一個縮小的image
		# if the current window does not meed our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

    # Python for迴圈 知識補充 :            
    # break：強制跳出 ❮整個❯ 迴圈
    # continue：強制跳出 ❮本次❯ 迴圈，繼續進入下一圈
    # pass：不做任何事情，所有的程式都將繼續

    # THIS IS WHERE WE WOULD PROCESS THE WINDOW, EXTRACT HOG FEATURES, AND  APPLY A MACHINE LEARNING CLASSIFIER TO PERFORM OBJECT 
    # DETECTION, since we do not have a classifier yet, let's just draw the window

        # 2.4 手動增生window : 
		# THIS IS WHERE WE WOULD PROCESS THE WINDOW, EXTRACT HOG FEATURES, AND
		# APPLY A MACHINE LEARNING CLASSIFIER TO PERFORM OBJECT DETECTION

		# since we do not have a classifier yet, let's just draw the window
		clone = layer.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)

		# normally we would leave out this line, but let's pause execution
		# of our script so we can visualize the window
		cv2.waitKey(1)
        # import time : 控制slide window速度library 
		time.sleep(0.05)