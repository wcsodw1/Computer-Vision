# python 1_1_David_detect_face.py

# Summary : 
    # 1.Detect image
    # 2.save(imwrite) bondingbox_image to File
    
# API : 1.cv2.waitKey(0) # Visualize the image until 設定手動關閉Visualize Image    


# 1. import the necessary packages
import cv2
image4 = cv2.imread("../../../Dataset/data/basketball.jpg")
image4 = cv2.resize(image4, (500, 400), interpolation=cv2.INTER_CUBIC)
    ## visualize
cv2.imshow("Faces", image4)
cv2.waitKey(0)


# 2. load our image and convert it to grayscale
gray = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
    ## visualize
cv2.imshow("Faces", gray)
cv2.waitKey(0)



# 3. Draw the Bondingbox

    # 3.1  load the face detector 
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
   
    # 3.2 detect faces in the image
rects = detector.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 9,
	minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the faces and draw a rectangle surrounding each
for (x, y, w, h) in rects:
	cv2.rectangle(image4, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #做一個有bonding box的灰階圖
for (x, y, w, h) in rects:
	cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # visualize
cv2.imshow("Faces", image4)
cv2.waitKey(0)
cv2.imshow("Faces", gray)
cv2.waitKey(0) # Visualize the image until手動把它關閉
cv2.imwrite("../../data/imwrite/chp1_1/basketball_box.jpg", image4) 
cv2.imwrite("../../data/imwrite/chp1_1/basketball_gray_box.jpg", gray) 
