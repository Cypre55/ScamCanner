from logreg import logr
import cv2
import numpy as np
import os
from transform import make_document
from sklearn import svm
import matplotlib.pyplot as plt
import math
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

def findy(r,t,x):
    return(int((r-x*math.cos(t))/math.sin(t)))

def findx(r,t,y):
    return(int((r-y*math.sin(t))/math.cos(t)))

def intersection(r,t,r1,t1):
    a=np.array([[math.cos(t),math.sin(t)],[math.cos(t1),math.sin(t1)]])
    det=np.linalg.det(a)
    if det<0.001:
        return None
    else:
        x=(r*math.sin(t1)-r1*math.sin(t))/det
        y=(-r*math.cos(t1)+r1*math.cos(t))/det
        return int(x),int(y)

def dist(pt1,pt2):
    return math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image,1
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized,1/r


def HoughLines(edges,x_max,y_max):
    theta_max = 1.0 * math.pi
    theta_min = -1.0 * math.pi / 2.0
    r_min = 0.0
    r_max = math.hypot(x_max, y_max)
    r_dim = 200
    theta_dim = 300
    hough_space = np.zeros((r_dim,theta_dim))
    for edge in edges:
        for itheta in range(theta_dim):
            x,y=edge.pt
            theta = 1.0 * itheta * (theta_max - theta_min) / theta_dim + theta_min
            r = x * math.cos(theta) + y * math.sin(theta)
            ir = r_dim * ( 1.0 * r ) / r_max
            hough_space[round(ir),round(itheta)] = hough_space[round(ir),round(itheta)] + 1

    neighborhood_size = 20
    threshold = 20

    data_max = filters.maximum_filter(hough_space, neighborhood_size)
    maxima = (hough_space == data_max)
    data_min = filters.minimum_filter(hough_space, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    temp=[]
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2
        if x_center>20 and x_center<theta_dim-20 and y_center<r_dim-20:
            temp.append((hough_space[int(y_center)][int(x_center)],x_center,y_center))

    temp=sorted(temp,reverse=True)
    print(len(temp))
    x,y=[],[]
    r=[]
    theta=[]
    for _,i,j in temp:
        b=True
        for x1,y1 in zip(x,y):
            if dist((x1,y1),(i,j))<10:
                b=False
        if b:
            x.append(i)
            y.append(j)
            r.append((1.0 * j * r_max ) / r_dim)
            theta.append(1.0 * i * (theta_max - theta_min) / theta_dim + theta_min)

    plt.imshow(hough_space, origin='lower')
    plt.plot(x,y, 'ro')
    cnt=1
    for i,j in zip(x,y):
        plt.text(i+1,j+1,str(cnt))
        cnt+=1
    plt.savefig('hough_space_maximas.png', bbox_inches = 'tight')
    plt.close()

    return r,theta

inputDir='TEST'
lr=logr()
surf = cv2.xfeatures2d.SURF_create(75)

for file in os.listdir(inputDir):
    img_path = './' + inputDir + '/' + file
    image=cv2.imread(img_path)
    org=image.copy()
    if(image.shape[0] > image.shape[1]):
        ratio=image.shape[0]/720
        image,ratio = image_resize(image, height=720)
    else:
        ratio=image.shape[1]/720
        image,ratio = image_resize(image, width=720)
        
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
    edges=[]
    nedges=[]
    for i in range(len(kps)):
        prob=lr.pred(descs[i].reshape(1,-1))
        if prob==1:
            edges.append(kps[i])
        else:
            nedges.append(kps[i])

    rem=[]
    left=[]
    kernel_size = 25
    thre = 3
    for i in range(len(edges)):
        x = (edges[i]).pt[0]
        y = (edges[i]).pt[1]
        count = 0
        for j in range(len(edges)):
            if j==i:
                continue
            if dist(edges[j].pt,(x,y))<=kernel_size:
                count += 1
        if count<thre:
            rem.append(edges[i])
        else:
            left.append(edges[i])
    edges=left

    x_max = image.shape[1]
    y_max = image.shape[0]
    d,theta = HoughLines(edges,x_max,y_max)#,min_dist=0)#,min_angle=0,threshold=0,num_peaks=20)
    print(len(d))
    print(d,theta)
    corners=[]
    d1=d[:4]
    theta1=theta[:4]
    for r,t in zip(d1,theta1):
        for r1,t1 in zip(d1,theta1):
            if r==r1 and t==t1:
                continue
            pt=intersection(r,t,r1,t1)
            if pt!=None and pt[0]>=0 and pt[0]<x_max and pt[1]>=0 and pt[1]<y_max:
                corners.append(pt)
    cont=[]
    for i in range(len(corners)):
        temp=[]
        temp.append(corners[i][0])
        temp.append(corners[i][1])
        temp=np.asarray(temp)
        temp1=[]
        temp1.append(temp)
        temp1=np.asarray(temp1)
        cont.append(temp)
    cont=np.asarray(cont)

    hull=[]

    try:
        hull.append(cv2.convexHull(cont,False))
        out_img = np.copy(image)
        cv2.drawKeypoints(image, edges, out_img, (0,0,255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.drawKeypoints(image, nedges, out_img, (255,0,0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.drawKeypoints(image, rem, out_img, (0,255,0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.drawContours(out_img, hull, 0, (255,255,0),1)
        cnt=1
        for r,t in zip(d,theta):
            s=math.sin(t)
            c=math.cos(t)
            x=r*c
            y=r*s
            x1=x+100000*s
            y1=y-100000*c
            x2=x-100000*s
            y2=y+100000*c
            color=(0,255,255)
            if cnt<=4:
                color=(0,102,255)
            cv2.line(out_img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            cnt+=1
        warped = make_document(org, hull, ratio)
        if(warped.shape[0] > warped.shape[1]):
                warp,_ = image_resize(warped, height=1000)
        else:
                warp,_ = image_resize(warped, width=1000)
        cv2.imshow("doc", warp)
        cv2.imshow("window", out_img)
        cv2.waitKey(0)
    except Exception as e:
        print(e)
    
    cv2.destroyAllWindows()
