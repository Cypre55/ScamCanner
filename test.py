from logreg import logr
import cv2
import numpy as np
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
lr=logr()
surf = cv2.xfeatures2d.SURF_create()

image=cv2.imread("Test.jpg")
if(image.shape[0] > image.shape[1]):
	image = image_resize(image, height=720)
else:
	image = image_resize(image, width=720)
#image = applyFilter(image, 50)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#x=max(image.shape[0],image.shape[1])
#s=1000.0/x
#image=cv2.resize(image,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
#image = cv2.medianBlur(image, 17)                             #UNCOMMENT THIS PART
#kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#image = cv2.filter2D(image, -1, kernel)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

kps, descs = surf.detectAndCompute(gray, None)
edges=[]
nedges=[]
edges1=[]
for i in range(len(kps)):
	prob=lr.pred(descs[i].reshape(1,-1))
	if(prob[0][1]>0.9):
		edges.append(kps[i].pt)
		edges1.append(kps[i])
	else:
		nedges.append(kps[i])

hull=cv2.convexHull(np.array(edges,dtype='float32'))
print(hull)
boundRect = cv2.boundingRect(np.array(hull,dtype='float32'))
cv2.rectangle(image,(int(boundRect[0]),int(boundRect[1])),(int(boundRect[2]),int(boundRect[3])),(0,0,255),3)
#cv2.drawContours(image,hull,0,(255, 0, 0), 1,8)
cv2.imshow('Hull',image)
ikps=np.empty((image.shape[0],image.shape[1],3),dtype=np.uint8)
cv2.drawKeypoints(image,nedges,ikps,color=(255,0,0))
cv2.drawKeypoints(ikps,edges1,ikps,color=(0,0,255))
cv2.imshow('SURF',ikps)
cv2.waitKey(0)
cv2.destroyAllWindows()
