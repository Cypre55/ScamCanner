import cv2
import numpy as np
import os
import csv

refPt = []
cropping = False
topN = 100
inputDir="ImageData"

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def click_and_crop(event, x, y, flags, param):
	global refPt, cropping
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))
		cropping = True
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cropping = False
		cv2.rectangle(ikps, refPt[-2], refPt[-1], (0, 255, 0), 2)
		cv2.imshow("image", ikps)

surf = cv2.xfeatures2d.SURF_create(75)
with open('Data4Training.csv', 'w') as f:
	writer=csv.writer(f)
	for file in os.listdir(inputDir):
		if file[-4:]!='.jpg':
			continue
		print("{} is loaded.".format(file))
		refPt=[]
		img_path = './' + inputDir + '/' + file
		image=cv2.imread(img_path)
		if(image.shape[0] > image.shape[1]):
			image = image_resize(image, height=720)
		else:
			image = image_resize(image, width=720)
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kernel = np.ones((7,7), np.float32)
		gray=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
		kernel_sharpening = np.array([[-1,-1,-1], 
	                              [-1, 9,-1],
	                              [-1,-1,-1]])
		sharp = cv2.filter2D(gray, -1, kernel_sharpening)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(sharp)
		kps, descs = surf.detectAndCompute(cl1, None)
		ikps=np.empty((image.shape[0],image.shape[1],3),dtype=np.uint8)
		img=image.copy()
		cv2.drawKeypoints(image,kps,ikps,color=(0,102,255))
		clone=ikps.copy()
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", click_and_crop)
		while True:
			cv2.imshow("image", ikps)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("r"):
				ikps= clone.copy()
				refPt = []
			elif key == ord("c"):
				break
		
		data=[]
		print(len(refPt))
		for j in range(len(kps)):
			temp=[]
			b=False
			for i in range(0,len(refPt),2):
				y1=min(refPt[i][1],refPt[i+1][1])
				y2=max(refPt[i][1],refPt[i+1][1])
				x1=min(refPt[i][0],refPt[i+1][0])
				x2=max(refPt[i][0],refPt[i+1][0])
				x=kps[j].pt[0]
				y=kps[j].pt[1]
				if x>x1 and x<x2 and y>y1 and y<y2:
					b=True
					break
			if b:
				temp.append(1)
				temp.append(kps[j])
				for k in range(64):
					temp.append(descs[j][k])
				data.append(temp)
				cv2.drawKeypoints(img,[kps[j]],img,color=(0,102,255))
			elif(j < topN):
				temp.append(0)
				temp.append(kps[j])
				for k in range(64):
					temp.append(descs[j][k])
				data.append(temp)
		cv2.imshow('corners',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		writer.writerows(data)
		os.remove(img_path)