import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import pickle
import joblib
import csv # TODO: Remove
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV


class logr:
	# Will search for the trainned model, if not found will generate it.
	def __init__(self):
		try:
			self.logmodel= joblib.load('trainned.pkl')
			print("Found the trainned data")
		except:
			self.make()
			print("Making the trainned data")

	# Will train the logistic regression
	def make(self):
		d=pd.read_csv("data.csv")
		d.drop("keypoint",axis=1,inplace=True)
		y=d["label"]
		x=d.drop("label",axis=1)
			
		os=SMOTE(random_state=0)
		col=x.columns
		os_data_x,os_data_y=os.fit_sample(x,y)
		os_data_x=pd.DataFrame(data=os_data_x,columns=col)
		os_data_y=pd.DataFrame(data=os_data_y,columns=['label'])
		x_train, x_test, y_train, y_test=train_test_split(os_data_x,os_data_y,test_size=0.25,random_state=100)

		self.logmodel=LogisticRegression(max_iter=500, C=849.75343591)
		self.logmodel.fit(x_train,y_train.values.ravel())
		predictions=self.logmodel.predict(x_test)

		print(classification_report(y_test,predictions))
		print(confusion_matrix(y_test,predictions))
		print(accuracy_score(y_test,predictions))

		joblib.dump(self.logmodel, 'trainned.pkl')
		
	def pred(self,desc):
		return self.logmodel.predict_proba(desc)

