# python David_2_1_template_matching.py --source "../../../CV_PyImageSearch/Dataset/data/source_01.jpg" --template "../../../CV_PyImageSearch/Dataset/data/template.jpg"
# python David_2_1_template_matching.py --source "../../../CV_PyImageSearch/Dataset/data/AUS.JPG" --template "../../../CV_PyImageSearch/Dataset/data/AUS_FACE.JPG"

    # import the necessary packages
import argparse
import cv2

    # (!!) 新增一張圖片參數 
    # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path to the source image")
ap.add_argument("-t", "--template", required=True, help="Path to the template image")
args = vars(ap.parse_args())

# 1.load the source and object(template) image
source = cv2.imread(args["source"])
source = cv2.resize(source,(400,600), interpolation = cv2.INTER_CUBIC)
template = cv2.imread(args["template"])
(tempH, tempW) = template.shape[:2]  # 範例(Object image)中RGB的某一層分別存入tempH, tempW
print("tempH, tempW : ",tempH, tempW) # 164 192
print("tempH, tempW type: ",type(tempH), type(tempW)) # int 


# 2.find the template in the source image

    # cv2.matchTemplate(source, Object, cv2.TM_CCOEFF)?
    
'''  
result = cv2.matchTemplate(image, templ, method[, result]) 參數
image :  被尋找的圖片 必須為 8-bit or 32-bit
templ :  尋找的物品圖片
         size不能大於 image，且格式需一致
method : 比對的方法
result : 比較的結果，格式為 numpy.ndarray (dtype=float32) 
         可傳入想儲存結果的 array 
         因 image 大小為W*H  且 templ 為w*h  ，所以大小為 (W-w+1)*(H-h+1)
'''

result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF)
print("result : ", result) # 164 192
print("result type: ",type(tempH), type(tempW)) # int 

(minVal, maxVal, minLoc, (x, y)) = cv2.minMaxLoc(result)

    # 抓出 object範圍 :  draw the bounding box on the source image
cv2.rectangle(source, (x, y), (x + tempW, y + tempH), (0, 255, 0), 2)

    # show the images
cv2.imshow("Source", source)
cv2.imshow("template", template)
cv2.waitKey(0)