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
from skimage.transform import hough_line_peaks

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

def HoughLines(edges,x_max,y_max,min_dist=5,min_angle=0,threshold=0.3,num_peaks=50):
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

    angles=np.arange(0,theta_dim)
    dists=np.arange(0,r_dim)
    print(min_dist)
    hspace,angles,dists=hough_line_peaks(hough_space,angles,dists,min_dist,min_angle,threshold*np.amax(hough_space),num_peaks)
    
    plt.imshow(hough_space,origin='lower')
    plt.plot(angles,dists, 'ro')
    plt.savefig('hough_space_maximas.png', bbox_inches = 'tight')
    plt.close()
    
    r=[]
    theta=[]
    for i,j in zip(dists,angles):
        r.append((1.0 * i * r_max ) / r_dim)
        theta.append(1.0 * j * (theta_max - theta_min) / theta_dim + theta_min)
    return r,theta

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
    corners=[]
    for r,t in zip(d,theta):
        for r1,t1 in zip(d,theta):
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
            x=[]
            y=[]
            s=math.sin(t)
            c=math.cos(t)
            if s<0.01 and s>-0.01:
                x.append(int(r/c))
                x.append(int(r/c))
                y.append(0)
                y.append(y_max-1)
            elif c<0.01 and c>-0.01:
                x.append(0)
                x.append(x_max-1)
                y.append(int(r/s))
                y.append(int(r/s))
            else:
                for x1 in (0,x_max-1):
                    y1=findy(r,t,x1)
                    if y1>=0 and y1<y_max:
                        x.append(x1)
                        y.append(y1)
                for y1 in (0,y_max-1):
                    if y1 in y:
                        continue
                    x1=findx(r,t,y1)
                    if x1>=0 and x1<x_max:
                        x.append(x1)
                        y.append(y1)
            #print(x,y,x_max,y_max)
            #if len(x)<2:
                #print("#####",r,t)
            if len(x)==2:
                cv2.line(out_img,(x[0],y[0]),(x[1],y[1]),(0,255,255),2)
                a,b=int(x[0]/2+x[1]/2),int(y[0]/2+y[1]/2)
                cv2.putText(out_img,str(cnt),(a,b),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,102),2)
                cnt+=1

        '''warped = make_document(org, hull, ratio)
        if(warped.shape[0] > warped.shape[1]):
                warp,_ = image_resize(warped, height=1000)
        else:
                warp,_ = image_resize(warped, width=1000)
        cv2.imshow("doc", warp)'''
        cv2.imshow("window", out_img)
        cv2.waitKey(0)
    except Exception as e:
        print(e)
    
    cv2.destroyAllWindows()
