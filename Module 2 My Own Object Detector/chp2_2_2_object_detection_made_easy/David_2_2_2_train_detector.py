# python David_2_2_2_train_detector.py --class "../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_stop_sign/stop_sign_images --annotations ../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_stop_sign/stop_sign_annotations --output ../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_stop_sign/output/2020_StopSignTest.svm'
# import the necessary packages
from __future__ import print_function
from imutils import paths
from scipy.io import loadmat
from skimage import io
import argparse
import dlib
import sys

# handle Python 3 compatibility
if sys.version_info > (3,):
	long = int

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True,
	help="Path to the CALTECH-101 class images")
ap.add_argument("-a", "--annotations", required=True,
	help="Path to the CALTECH-101 class annotations")
ap.add_argument("-o", "--output", required=True,
	help="Path to the output detector")


# import sys 
# sys.argv[1:] = '-c stop_sign_images -a stop_sign_annotations -o output/stop_sign_detector.svm'.split()
# sys.argv[1:] = '-c Airplane/image -a Airplane/annotations -o Airplane/output/airplane.svm'.split()
sys.argv[1:] = '-c ../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_stop_sign/stop_sign_images -a ../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_stop_sign/stop_sign_annotations -o ../../../CV_PyImageSearch/Dataset/Chapter_Specific/chp2_2_stop_sign/output/2020_StopSignTest.svm'.split()

args = vars(ap.parse_args())


# grab the default training options for our HOG + Linear SVM detector initialize the
# list of images and bounding boxes used to train the classifier
print("[INFO] gathering images and bounding boxes...")
options = dlib.simple_object_detector_training_options()
images = []
boxes = []


# loop over the image paths
for imagePath in paths.list_images(args["class"]):
    # extract the image ID from the image path and load the annotations file
    imageID = imagePath[imagePath.rfind("/") + 1:].split("_")[1]
    #print(imageID)
    id1= imagePath.find("\\")
    #print(id1)
    id2= imagePath[id1+1:]
    #print(id2)
    imageID = id2.replace(".jpg", "")
    #print(imageID)
    str = imageID[6:]
    #print(str)
    #dir= "./Airplane1/annotations/"
    dir= "./stop_sign_annotations/"
    #print(dir)
    p = "{}annotation_{}.mat".format(dir, str)
    #print(p)
    annotations = loadmat(p)["box_coord"]
    #print(annotations)
    bb = [dlib.rectangle(left=long(x), top=long(y), right=long(w), bottom=long(h)) 
        for (y, h, x, w) in annotations]
    #print(bb)
    boxes.append(bb)    
    #print(boxes)
    #print(len(boxes))
    # add the image to the list of images
    images.append(io.imread(imagePath))
    #print(images)

# train the object detector
print("[INFO] training detector...")
detector = dlib.train_simple_object_detector(images, boxes, options)

# dump the classifier to file
print("[INFO] dumping classifier to file...")
detector.save(args["output"])

# visualize the results of the detector
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()