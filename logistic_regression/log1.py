import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.stats import boxcox
import imblearn

l = range(2,64)
y_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = [0])
x_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = l)
y_df = np.array(y_df[0])
X_train, X_test, Y_train, Y_test = train_test_split(x_df, y_df, test_size = 0.3, random_state = 0)

#   Simple logistic regression

log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
a = log_reg.score(X_test, Y_test)*100
print("Accuracy using logistic regression = {}".format(a))

#   Using SMOTE for over sampling

ovr = imblearn.over_sampling.SMOTE()
X_train_n, Y_train_n = ovr.fit_resample(X_train, Y_train)
log_reg.fit(X_train_n, Y_train_n)
X_test_n, Y_test_n = ovr.fit_resample(X_test, Y_test)
a = log_reg.score(X_test_n, Y_test_n)*100
print("Accuracy using logistic regression (Using SMOTE) = {}".format(a))

#   SVM with class weights

S1 = svm.SVC(C =5000.0, class_weight = {1:1.63, 0:1.0})
S1.fit(X_train, Y_train)
a = S1.score(X_test, Y_test)*100
print("Accuracy using SVM (with class weights) = {}".format(a))

#   Simple SVM

S1 = svm.SVC()
S1.fit(X_train, Y_train)
a = S1.score(X_test, Y_test)*100
print("Accuracy using SVM = {}".format(a))

