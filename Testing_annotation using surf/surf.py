import cv2
import numpy as np
import csv
# Reading test image
img = cv2.imread("test.jpg",0)
out_img = np.copy(img)
# Running surf detector
surf = cv2.xfeatures2d.SURF_create(7000)
key_points, descriptors = surf.detectAndCompute(img,None)
# Manually figured out corner points 
dim = descriptors.shape
corner = [0,2,5,9]
# Writing data in csv file
f_data = []
for i in range(len(key_points)):
    temp = []
    if i in corner:
        temp.append(1)
    else:
        temp.append(0)
    temp.append(key_points[i])
    for j in range(dim[1]):
        temp.append(descriptors[i][j])
    f_data.append(temp)
with open('information.csv','w',newline = '')  as file:
    writer = csv.writer(file)
    writer.writerows(f_data)
# Drawing points
cv2.drawKeypoints(img, key_points, out_img, (0,0,0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
cv2.imshow("window", out_img)
cv2.waitKey(10000)