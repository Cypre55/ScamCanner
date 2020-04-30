import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import joblib

class logr:
	def __init__(self):
		try:
			self.logmodel= joblib.load('trainned.pkl')
			print("Found the trainned data")
		except:
			self.make()
			print("Making the trainned data")
	def make(self):
		d=pd.read_csv("data.csv")
		d.drop("keypoint",axis=1,inplace=True)
		y=d["label"]
		x=d.drop("label",axis=1)
			
		x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=100)

		self.logmodel=LogisticRegression(solver='saga')
		self.logmodel.fit(x_train,y_train.values.ravel())
		predictions=self.logmodel.predict(x_test)

		print(classification_report(y_test,predictions))
		print(confusion_matrix(y_test,predictions))
		print(accuracy_score(y_test,predictions))

		joblib.dump(self.logmodel, 'trainned.pkl')
		
	def pred(self,desc):
		return self.logmodel.predict(desc)

