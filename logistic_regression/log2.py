import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.stats import boxcox
import imblearn
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os, subprocess
import math

def weight(dis, var):
    t = (-1)*(dis**2)/(2*var)
    return math.exp(t)

l = range(2,66)
y_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = [0])
x_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = l)
x_normalized = preprocessing.normalize(x_df, norm='l2')
y_df = np.array(y_df[0])
log_reg = LogisticRegression(solver = 'saga', max_iter = 500, C = 100.0)
log_reg.fit(x_df, y_df)

inputDir = "ImageData"

surf = cv2.xfeatures2d.SURF_create(300)
for file in os.listdir(inputDir):
    if file.endswith(".jpg"):
        print("{} is loaded.".format(file))
        img_path = inputDir + '/' + file
        img1 = cv2.imread(img_path,1)
        img1 = cv2.medianBlur(img1,29)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img1 = cv2.filter2D(img1, -1, kernel)
        img2 = cv2.filter2D(img1, -1, kernel)
        cv2.addWeighted(img1, 2, img2, -1, 0, img1)
        scale_percent = 25
        width = int(img1.shape[1] * scale_percent / 100)
        height = int(img1.shape[0] * scale_percent / 100)
        dim = (width, height)
        img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = surf.detectAndCompute(img,None)
        dim = descriptors.shape
        key_points_e = []
        key_points_ne = []
        descriptors = pd.DataFrame(descriptors)

        a = log_reg.predict(descriptors)
        for i in range(len(a)):
            if a[i]==1:
                key_points_e.append(key_points[i])
            else:
                key_points_ne.append(key_points[i])
        # kernel_size = 9
        # thre = 10
        # for i in range(len(key_points_e)):
        #     if i>=len(key_points_e):
        #         break
        #     x = (key_points_e[i]).pt[0]
        #     y = (key_points_e[i]).pt[1]
        #     count = 0
        #     for j in range(len(key_points_e)):
        #         if j==i:
        #             continue
        #         if abs((key_points_e[j]).pt[0]-x) <= kernel_size and abs((key_points_e[j]).pt[1]-y) <= kernel_size:
        #             count += 1
        #     if count <= thre:
        #         key_points_ne.append(key_points_e[i])
        #         key_points_e.pop(i)
        cont = []
        for i in range(len(key_points_e)):
            temp = []
            temp.append(int((key_points_e[i]).pt[0]))
            temp.append(int((key_points_e[i]).pt[1]))
            temp = np.asarray(temp)
            temp1 = []
            temp1.append(temp)
            temp1 = np.asarray(temp1)
            cont.append(temp)
        cont = np.asarray(cont)
        hull = []
        hull.append(cv2.convexHull(cont,False))
        rect = cv2.minAreaRect(hull[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = rect[1][0]
        height = rect[1][1]
        temp = (min(width, height))/2
        # print(box)
        for i in range(len(box)):
            t1 = 0
            t2 = 0
            count = 0
            for j in range(len(key_points_e)):
                p = (key_points_e[j]).pt
                point2 = [p[0],p[1]]
                dis = (((box[i][0] - point2[0])**2 + (box[i][1] - point2[1])**2)**0.5)
                if dis < temp:
                    t1 += point2[0] * weight(dis, 10000)
                    t2 += point2[1] * weight(dis, 10000)
                    count += weight(dis, 10000)
            if count==0:
                continue
            box[i][0] = t1/count
            box[i][1] = t2/count
        # print(box)
        out_img = np.copy(img1)
        cv2.drawKeypoints(img1, key_points_e, out_img, (0,0,255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.drawKeypoints(img1, key_points_ne, out_img, (255,0,0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.drawContours(out_img, hull, 0, (255,255,0), 1)
        cv2.drawContours(out_img, [box], 0, (0,0,255),8)
        cv2.imshow("window", out_img)
        cv2.waitKey(10000)
            