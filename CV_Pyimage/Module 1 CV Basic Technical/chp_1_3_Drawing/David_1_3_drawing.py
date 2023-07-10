# -*- coding: utf-8 -*-

# python David_1_3_drawing.py -i "../../../CV-PyImageSearch Gurus Course/Dataset/data/AUS.JPG" -i2 "../../../CV-PyImageSearch Gurus Course/Dataset/data/AUS1.JPG"
"""
Created on Fri Mar 20 23:10:18 2020

@author: user
"""

import numpy as np
import cv2
import argparse

# load the tic-tac-toe image and convert it to grayscale
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-i2", "--image2", required=True, help="Path to the image2")

args = vars(ap.parse_args())

Aus = cv2.imread(args["image"])
print(Aus.shape)
Aus = cv2.resize(Aus, (480, 550), interpolation=cv2.INTER_CUBIC)
print(Aus.shape)


cv2.circle(Aus, (185, 150), 60, (0, 0, 255), 2)
cv2.circle(Aus, (163, 154), 10, (0, 0, 255), 3)
cv2.circle(Aus, (203, 153), 10, (0, 0, 255), 3)
cv2.rectangle(Aus, (167, 178), (202, 195), (0, 0, 255),1)

cv2.imshow("Australia", Aus)
cv2.waitKey(0)


#cv2.imread("florid_trip.png")
#cv2.waitKey(0)


canvas = cv2.imread(args["image2"])
print(canvas.shape)

canvas = cv2.resize(canvas, (800, 600), interpolation=cv2.INTER_CUBIC)

# draw a green line from the top-left corner of our canvas to the
# bottom-right
green = (0, 255, 0)
cv2.line(canvas, (0, 0), (800, 600), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# now, draw a 3 pixel thick red line from the top-right corner to the
# bottom-left
red = (0, 0, 255)
cv2.line(canvas, (800, 0), (0, 600), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw a green 50x50 pixel square, starting at 10x10 and ending at 60x60
cv2.rectangle(canvas, (270, 350), (480, 600), green,5)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw another rectangle, this time we'll make it red and 5 pixels thick
cv2.rectangle(canvas, (280, 360), (470, 590), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# let's draw one last rectangle: blue and filled in
blue = (255, 0, 0)
cv2.rectangle(canvas, (380, 20), (530, 90), blue, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# reset our canvas and draw a white circle at the center of the canvas with
# increasing radii - from 25 pixels to 150 pixels

canvas = cv2.imread(args["image2"])
print(canvas.shape)
canvas = cv2.resize(canvas, (800, 600), interpolation=cv2.INTER_CUBIC)

(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

for r in range(2, 300, 20):
	cv2.circle(canvas, (centerX, centerY), r, white)

# Show our masterpiece
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

