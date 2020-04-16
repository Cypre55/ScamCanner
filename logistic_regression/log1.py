import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.stats import boxcox

l = range(2,64)
y_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = [0])
x_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = l)
y_df = np.array(y_df[0])
X_train, X_test, Y_train, Y_test = train_test_split(x_df, y_df, test_size = 0.3, random_state = 0)
log_reg = LogisticRegression(class_weight = {1:1.63, 0:1.0})
log_reg.fit(X_train, Y_train)
a = log_reg.score(X_test, Y_test)*100
print("Accuracy using logistic regression = {}".format(a))

S1 = svm.SVC(C =5000.0, class_weight = {1:1.63, 0:1.0})
S1.fit(X_train, Y_train)
a = S1.score(X_test, Y_test)*100
print("Accuracy using SVM (with class weights) = {}".format(a))

S1 = svm.SVC()
S1.fit(X_train, Y_train)
a = S1.score(X_test, Y_test)*100
print("Accuracy using SVM = {}".format(a))