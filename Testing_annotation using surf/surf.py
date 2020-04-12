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
key_points_filtered = []
key_points_filtered.append(key_points[0])
key_points_filtered.append(key_points[2])
key_points_filtered.append(key_points[5])
key_points_filtered.append(key_points[9])
descriptors_f = []
descriptors_f.append(descriptors[0])
descriptors_f.append(descriptors[2])
descriptors_f.append(descriptors[5])
descriptors_f.append(descriptors[9])
f_data = []
# Writing data in csv file
for i in range(len(key_points_filtered)):
    temp = []
    temp.append(key_points_filtered[i])
    for j in range(dim[1]):
        temp.append(descriptors_f[i][j])
    f_data.append(temp)
with open('information.csv','w',newline = '')  as file:
    writer = csv.writer(file)
    writer.writerows(f_data)
# Drawing points
cv2.drawKeypoints(img, key_points_filtered, out_img, (0,0,0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
cv2.imshow("window", out_img)
cv2.waitKey(10000)