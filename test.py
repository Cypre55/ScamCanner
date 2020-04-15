from logreg import logr
import cv2
import numpy as np

lr=logr()
surf = cv2.xfeatures2d.SURF_create()

image=cv2.imread("test.jpg")
x=max(image.shape[0],image.shape[1])
s=1000.0/x
image=cv2.resize(image,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
kps, descs = surf.detectAndCompute(gray, None)
edges=[]
nedges=[]
for i in range(len(kps)):
	prob=lr.pred(descs[i].reshape(1,-1))
	if(prob[0][1]>0.9):
		edges.append(kps[i])
	else:
		nedges.append(kps[i])

ikps=np.empty((image.shape[0],image.shape[1],3),dtype=np.uint8)
cv2.drawKeypoints(image,nedges,ikps,color=(255,0,0))
cv2.drawKeypoints(ikps,edges,ikps,color=(0,0,255))
cv2.imshow('SURF',ikps)
cv2.waitKey(0)
cv2.destroyAllWindows()