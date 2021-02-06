import cv2
import numpy as np
import os
import csv

inputDir = "ImageData"

surf = cv2.xfeatures2d.SURF_create()
with open('information.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file in os.listdir(inputDir):
        if file[-4:] != '.jpg':
            continue
        print("{} is loaded.".format(file))
        img_path = './'+inputDir + '/' + file
        image = cv2.imread(img_path)
        image = cv2.resize(image, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 29)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(blur, -1, kernel)
        kps, descs = surf.detectAndCompute(sharp, None)
        l = 200
        kps = kps[l:l+100]
        ikps = np.empty((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        cv2.drawKeypoints(image, kps, ikps, color=(25, 255, 64))
        cv2.imshow("SURF", ikps)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        data = []
        for i in range(len(kps)):
            ikps = np.empty((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            cv2.drawKeypoints(image, [kps[i]], ikps, color=(25, 255, 64))
            cv2.imshow('SURF', ikps)
            print("Label(Space for Yes, any other fo No):- ")
            enter = cv2.waitKey(0)
            temp = []
            if enter == 32:
                temp.append(1)
            else:
                temp.append(0)
            print(i, temp)
            temp.append(kps[i])
            for j in range(64):
                temp.append(descs[i][j])
            data.append(temp)
        writer.writerows(data)

cv2.destroyAllWindows()
