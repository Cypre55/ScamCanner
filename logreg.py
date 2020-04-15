import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

class logr:
	def __init__(self):
		d=pd.read_csv("data.csv")
		d.drop("keypoint",axis=1,inplace=True)
		y=d["label"]
		x=d.drop("label",axis=1)
		x_train, x_test1, y_train, y_test1=train_test_split(x,y,test_size=0.25,random_state=1)
		os=SMOTE(random_state=0)
		col=x_train.columns
		os_data_x,os_data_y=os.fit_sample(x_train,y_train)
		os_data_x=pd.DataFrame(data=os_data_x,columns=col)
		os_data_y=pd.DataFrame(data=os_data_y,columns=['label'])
		x_train, x_test, y_train, y_test=train_test_split(os_data_x,os_data_y,test_size=0.25,random_state=5)
		self.logmodel=LogisticRegression()
		self.logmodel.fit(x_train,y_train.values.ravel())
		predictions=self.logmodel.predict(x_test)
		print(classification_report(y_test,predictions))
		print(confusion_matrix(y_test,predictions))
		print(accuracy_score(y_test,predictions))
	def pred(self,desc):
		return self.logmodel.predict_proba(desc)

