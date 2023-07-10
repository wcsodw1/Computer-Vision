# (Set your own Path and File name)python David_2_2_2_test_detector.py --detector "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_elephant/Detector/elephant_detector.svm" -t "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_elephant/TestImage"

# python David_2_2_2_test_detector.py -d output/20200724_Face_Detector.svm -t cebu

# 1. Preprocessing 
#(!!)  import dlib / from imutils import paths

    # import the necessary packages
from imutils import paths
import argparse
import dlib
import cv2
import sys 

    # 1.1 Set testing path
    # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="Path to trained object detector")
ap.add_argument("-t", "--testing", required=True, help="Path to directory of testing images")


sys.argv[1:] = '-d output/20200724_Face_Detector.svm -t cebu'.split()
#sys.argv[1:] = '-d Airplane\\output\\airplane.svm -t Airplane\\testing'.split()
# sys.argv[1:] = '-d Airplane1\\output\\airplane1.svm -t Airplane1\\testing'.split()

args = vars(ap.parse_args())

# 2. load the detector
detector = dlib.simple_object_detector(args["detector"])

# loop over the testing images
for testingPath in paths.list_images(args["testing"]):
	# load the image and make predictions
	image = cv2.imread(testingPath)
	print(image)
	boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	print(boxes)

	# loop over the bounding boxes and draw them
	for b in boxes:
		(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
		cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)