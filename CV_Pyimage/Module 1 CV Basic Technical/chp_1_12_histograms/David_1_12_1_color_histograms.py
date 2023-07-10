# USAGE
# python David_1_12_1_color_histograms.py --image beach.png
# python David_1_12_1_color_histograms.py --image "../../data/horseshoe_bend.png"
# python David_1_12_1_color_histograms.py --image "../../../CV-PyImageSearch Gurus Course/Dataset/data/horseshoe_bend.png"

# interpolation : https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/interpolation_methods.html

# 1. Preprocessing : 

    # 1.1 import the necessary packages
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

    # 1.2 construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
print("image:",image)
print(image.shape) # (475, 600, 3)
print(type(image)) # numpy array

# 2. Flattened' Color Histogram : 

    # 2.1 (!!) Figure 1 , Flattened' Color Histogram
        # 2.1.1 numpy -> List
chans = cv2.split(image)
print("chans:",chans)
print(type(chans))

    # 2.2 Add Plot describe (plo中 標題(title)及x, y軸的文字描述)
colors = ("B","G","R")  
plt.figure()
plt.title("Flatten-color-Histogram")
plt.xlabel("x axis(Bins)")
plt.ylabel("Y axis (# of Pixels)")

    # 2.3 (!!!) loop to visualize in the plot(用迴圈繪出圖形)
for (chan, i) in zip(chans, colors):  # 放到for裡面的chans是一個list的型態
     
	# create a histogram for the current channel and plot it
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color = i)
	plt.xlim([0, 256]) # 設定x軸範圍(一般來說都落在 0-255中)

# 3. 2D histograms : 
    # let's move on to 2D histograms -- we need to reduce the number of bins in the histogram from 256 to 32 so we can better visualize the results
fig = plt.figure()

    # 2.1.1 plot a 2D color histogram for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)

    # 2.1.2 plot a 2D color histogram for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)

    # 2.1.3 plot a 2D color histogram for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)


# 4. 2D Histogram and 3D Histogramshape
    
    #  4.1 (2D Hist shape 與 faltten shape ) : make sure the 2D Hist shape and the faltten shape values 
    # finally, let's examine the dimensionality of one of the 2D histograms
print("2D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0])) # 2D histogram shape: (32, 32), with 1024 values


    # 4.2  3D Histogram shape(只能show shape無法建3D圖) : 
    # our 2D histogram could only take into account 2 out of the 3 channels in the image so now let's build a 3D color histogram
    # (utilizing all channels) with 8 bins in each direction -- we can't plot the 3D histogram, but the theory is exactly like
    # that of a 2D histogram, so we'll just show the shape of the histogram

hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0])) # 3D histogram shape: (8, 8, 8), with 512 values

# display the image with matplotlib to avoid GUI conflicts on macOS
plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(image))

# Show our plots
plt.show()
plt.close()

#-------------------------- chp1_12_2 grayscale_histogram.py


# 1. Turn to grayscale

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("image : ", type(image))
print("image shape: ", image.shape)
cv2.imshow("image",image)

    # 1.1 (!!)construct a grayscale histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
print("hist : ", type(hist))
print("hist shape: ", hist.shape)
cv2.imshow("hist",hist)
cv2.waitKey(0)


    # matplotlib expects RGB images so convert and then display the image
    # with matplotlib to avoid GUI conflicts/errors (mainly on macOS)
plt.figure()
plt.axis("on") # x ,y axis reveal or not
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

    # plot the histogram
plt.figure()
plt.title("Grayscale-Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

    # 1.2(!!) normalize the histogram
    # plot the normalized histogram

hist /= hist.sum()
plt.figure()
plt.title("Grayscale Histogram (Normalized)")
plt.xlabel("x axis(Bins)")
plt.ylabel("Y axis - % of Pixels ")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
plt.close()

# ----------------------- chp1_12_3 equalize.py (均衡化)

    #(!!) cv2.equalizeHist()只提供灰度值處理 RGB匯報錯
    
    
# apply histogram equalization to stretch the constrast of our image
equalize = cv2.equalizeHist(image)
# show our images -- notice how the constrast of the second image has
# been stretched
cv2.imshow("Original", image)
cv2.imshow("Histogram Equalization", equalize)
cv2.waitKey(0)



# ------------------------------ chp1_12_4 histogram_with_mask.py 這張可以學習如何遮罩
import numpy as np


# grab the image channels, initialize the tuple of colors and the figure

def plot_histogram(image, title, mask = None) :
	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")

	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and plot it
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color=color)
		plt.xlim([0, 256])
        
image = cv2.imread(args["image"])
cv2.imshow("Original", image)
plot_histogram(image, "Histogram for Original Image")

    # construct a mask for our image -- our mask will be BLACK for regions
    # we want to IGNORE and WHITE for regions we want to EXAMINE
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (60, 290), (210, 390), 255, -1) # 設定mask區域
cv2.imshow("Mask", mask)

    # 原影像圖與mask結合 ,what does masking our image look like?
masked = cv2.bitwise_and(image, image, mask=mask) 
cv2.imshow("Applying the Mask", masked)

    # compute a histogram for our image, but we'll only include pixels in the masked region
plot_histogram(image, "Histogram for Masked Image", mask=mask)
plt.show()