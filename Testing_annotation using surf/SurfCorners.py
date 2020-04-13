

import numpy as np 
import cv2
import csv
#############  Parameters for tweeking  ################
topN  = 3 
c= open('scans.csv', 'w')
writer = csv.writer(c)
image = cv2.imread('Test.jpg',0)
image=cv2.resize(image,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
cv2.waitKey(0)
featureDetector = cv2.xfeatures2d.SURF_create()
(kps, descs) = featureDetector.detectAndCompute(image, None)

fields=[0,1]
kps = kps[:topN]
corners=[]
	
for i in range(len(kps)):
		
	points =[]
	points.append(kps[i])
	
	ikps = np.empty((image.shape[0], image.shape[1], 3), dtype = np.uint8)	
	cv2.drawKeypoints(image, points, ikps)
	cv2.imshow('SURF', ikps)
	cv2.waitKey(1000)
	enter=input("Corner?")
	if(enter==1):
		corners.append(i)
data=[]
for i in range(len(kps)):
	temp=[]
	if i in corners:
		temp.append(1)
	else:
		temp.append(0)
	temp.append(kps[i])
	temp2=descs[i,:]
	for j in range(64):
            temp.append(temp2[j])
	data.append(temp)
writer.writerows(data)


