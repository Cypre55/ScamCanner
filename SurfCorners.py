import numpy as np 
import cv2
import csv
import os, subprocess
#############  Parameters for tweeking  ################
topN  = 30
inputDir = "ARK scamcanner data"

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def applyFilter(image, sensitivity = 30):
	# img = cv2.medianBlur(image, 17)
	# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
	# img = cv2.filter2D(img, -1, kernel)

	# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# lower_white = np.array([0, 0, 255-sensitivity])
	# upper_white = np.array([180, sensitivity, 255])

	# filtered = cv2.inRange(img, lower_white, upper_white)
	# filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

	# img = cv2.GaussianBlur(image, (15, 15), 10 )
	# filtered = cv2.pyrMeanShiftFiltering(img, 70, 50, 5)
	
	filtered = image

	return filtered

with open('Data4Training.csv', 'w') as c:
	writer = csv.writer(c)
	for file in os.listdir(inputDir):
		if file.endswith(".jpg"):
			print("{} is loaded.".format(file))
			sourceFile = inputDir + '/' + file
		
			image = cv2.imread(sourceFile)
			if(image.shape[1] > image.shape[0]):
				image = image_resize(image, height=720)
			else:
				image = image_resize(image, width=720)
			# cv2.waitKey(0)

			imagef = applyFilter(image, 50)
			imageg = cv2.cvtColor(imagef, cv2.COLOR_BGR2GRAY)

			featureDetector = cv2.xfeatures2d.SURF_create()
			(kps, descs) = featureDetector.detectAndCompute(imageg, None)
		
			fields=[0,1]
			if(len(kps) > topN):
				kps = kps[:topN]
			corners=[]

			"""
			ikps = np.empty((image.shape[0], image.shape[1], 3), dtype = np.uint8)
			cv2.drawKeypoints(image, kps, ikps, (178, 0, 255))
			cv2.imshow('SURF', ikps)
			cv2.imshow('Image', imagef)
			cv2.waitKey(0)

			"""	
			for i in range(len(kps)):
					
				points =[]
				points.append(kps[i])
				
				ikps = np.empty((image.shape[0], image.shape[1], 3), dtype = np.uint8)	
				cv2.drawKeypoints(image, points, ikps, (178, 0, 255))
				cv2.imshow('SURF', ikps)
				cv2.waitKey(1)
				enter=int(input("{}) Corner (0 or 1) :".format(i+1)))
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
			
