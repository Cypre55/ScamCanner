import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy.stats import boxcox
import imblearn
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns

l = range(2,66)
y_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = [0])
x_df = pd.read_csv('information.csv', delimiter = ',', header = None, usecols = l)
x_normalized = preprocessing.normalize(x_df, norm='l2')
y_df = np.array(y_df[0])
X_train, X_test, Y_train, Y_test = train_test_split(x_normalized, y_df, test_size = 0.3, random_state = 0)

#  Simple logistic regression

log_reg = LogisticRegression(solver = 'saga', max_iter = 500, C = 100.0)
log_reg.fit(X_train, Y_train)
y_pre = log_reg.predict(X_test)
a = log_reg.score(X_test, Y_test)*100
conf = metrics.confusion_matrix(Y_test, y_pre)
print(conf)
print("Accuracy using logistic regression = {}".format(a))

# #  Using SMOTE for over sampling

# ovr = imblearn.over_sampling.SMOTE()
# X_train_n, Y_train_n = ovr.fit_resample(X_train, Y_train)
# log_reg.fit(X_train_n, Y_train_n)
# X_test_n, Y_test_n = ovr.fit_resample(X_test, Y_test)
# a = log_reg.score(X_test_n, Y_test_n)*100
# print("Accuracy using logistic regression (Using SMOTE) = {}".format(a))

# #   Simple SVM

# S1 = svm.SVC()
# S1.fit(X_train, Y_train)
# a = S1.score(X_test, Y_test)*100
# print("Accuracy using SVM = {}".format(a))

