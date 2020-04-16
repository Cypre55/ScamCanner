import cv2
import numpy as np
import os
import csv

inputDir = "Photos/Photos"

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
        sharp = cv2.Canny(sharp,100,200)
        l = 200
        kps = kps[l:l+100]
        # ikps = np.empty((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        # cv2.drawKeypoints(image, kps, ikps, color=(25, 255, 64))
        # sharp = cv2.medianBlur(sharp, 29)
        # sharp = cv2.filter2D(sharp, -1, kernel)
        # cv2.imshow("SURF", sharp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        data = []
        for i in range(len(kps)):
            x, y = kps[i].pt
            x = int(x)
            y = int(y)
            flag = 0
            for j in range(-5,5,1):
                for k in range(-5,5,1):
                    if sharp[y+j,x+k]>=245:
                        flag = 1

#             if flag == 1:
#                 cv2.circle(image, (x,y), 5, (0,255,0),-1)
#             else:
#                 cv2.circle(image, (x,y), 5, (255,0,0),-1)

            ikps = np.empty((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            # cv2.drawKeypoints(image, [kps[i]], ikps, color=(25, 255, 64))
            # print("Label(Space for Yes, any other fo No):- ")
            # enter = cv2.waitKey(0)
            temp = []
            temp.append(flag)
            # print(i, temp)
            temp.append(kps[i])
            for j in range(64):
                temp.append(descs[i][j])
            data.append(temp)
        # print("Do you want to add the data points?(y/n)\n")
        # cv2.imshow('SURF', image)
        # ch = cv2.waitKey(0)
        # if(ch == 'Y' or ch == 'y'):
        writer.writerows(data)
        #     print("{} is added".format(file))
        # else:
        #     print("{} not added". format(file))
cv2.destroyAllWindows()
