# python David_2_3_1_test_pyramid.py --image "../../data/florida_trip.png" --scale 1.5
#(Set your own File name & image path) python David_2_3_1_test_pyramid.py --image "../../../CV_PyImageSearch/Dataset/data/AUS.JPG" --scale 1.1


# 1.Preprocessing
    # import the necessary packages
from pyimagesearch.object_detection.helpers import pyramid
import argparse
import cv2

    # 1.1 (!!) 多加一行比例縮放參數 
    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())

    # load the input image
image = cv2.imread(args["image"]) 

# 2.(!!)迴圈跑到圖像小到不能再小收斂
# loop over the layers of the image pyramid and display them
for (i, layer) in enumerate(pyramid(image, scale=args["scale"])):
	cv2.imshow("Layer of image{}".format(i + 1), layer)
	cv2.waitKey(0)