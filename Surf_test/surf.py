import cv2
import numpy as np
import csv
import os, subprocess

############### Parameters to be tweaked ###############
inputDir = "ImageData"
topN = 100

# Running surf detector
surf = cv2.xfeatures2d.SURF_create(1000)
with open('information.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    for file in os.listdir(inputDir):
        if file.endswith(".jpg"):
            print("{} is loaded.".format(file))
            img_path = inputDir + '/' + file
            img1 = cv2.imread(img_path,1)

            img1 = cv2.medianBlur(img1,101)
            img1 = cv2.blur(img1, (11, 11))
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img1 = cv2.filter2D(img1, -1, kernel)
            img1 = cv2.filter2D(img1, -1, kernel)
            img1 = cv2.resize(img1, None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_AREA)

            img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            key_points, descriptors = surf.detectAndCompute(img,None)
            dim = descriptors.shape
            edge = []

            # Labelling of corner or edge points
            for i in range(topN):   
                out_img = np.copy(img)
                print(i)
                key_points_f = []
                key_points_f.append(key_points[i])
                # Drawing points
                out_img = np.empty((img1.shape[0], img1.shape[1], 3), dtype = np.uint8)
                cv2.drawKeypoints(img, key_points_f, out_img, (179,0,255))
                cv2.imshow("window", out_img)
                cv2.waitKey(200)
                a = int(input("label :- "))
                if a==1:
                    edge.append(i)
            # Writing data in csv file
            f_data = []
            for i in range(topN):
                temp = []
                if i in edge:
                    temp.append(1)
                else:
                    temp.append(0)
                temp.append(key_points[i])
                for j in range(dim[1]):
                    temp.append(descriptors[i][j])
                f_data.append(temp)
            writer.writerows(f_data)

