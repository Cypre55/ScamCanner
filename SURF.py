import numpy as np 
import cv2

#############  Parameters for tweeking  ################
topN  = 10000  # Top N features

image = cv2.imread("IMG_20200411_230312.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
featureDetector = cv2.xfeatures2d.SURF_create()
(kps, descs) = featureDetector.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

# Taking only the top N features
kps = kps[:topN]

ikps = np.empty((image.shape[0], image.shape[1], 3), dtype = np.uint8)
cv2.drawKeypoints(image, kps, ikps)

cv2.imshow('SURF', ikps)
cv2.waitKey(0)
cv2.imwrite('OUT.jpg', ikps)