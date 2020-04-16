import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.stats import boxcox
import imblearn
import os, subprocess

l = range(2,66)
y_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = [0])
x_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = l)
y_df = np.array(y_df[0])
S1 = svm.SVC(C =5000.0, class_weight = {1:1.63, 0:1.0})
S1.fit(x_df, y_df)

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

        a = S1.predict(descriptors)
        for i in range(len(a)):
            if a[i]==1:
                key_points_e.append(key_points[i])
            else:
                key_points_ne.append(key_points[i])
        out_img = np.copy(img1)
        cv2.drawKeypoints(img1, key_points_e, out_img, (0,0,255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.drawKeypoints(img1, key_points_ne, out_img, (255,0,0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.imshow("window", out_img)
        cv2.waitKey(10000)
            