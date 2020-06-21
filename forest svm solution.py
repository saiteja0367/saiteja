# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:49:33 2019

@author: sai teja
"""

import pandas as pd
import numpy as np
forest=pd.read_csv("file:///C:/Users/asdf/Desktop/assignments/svm/forestfires.csv")
forest.drop(["month"],inplace=True,axis=1)
forest.drop(["day"],inplace=True,axis=1)
forest["size_category"]=lb.fit_transform(forest["size_category"])
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest,test_size=0.3)
train_x=train.iloc[:,0:28]
train_y=train.iloc[:,28]
test_x=train.iloc[:,0:28]
test_y=train.iloc[:,28]
from sklearn.svm import SVC
#kernel=linear
model_linear=SVC(kernel="linear")
model_linear.fit(train_x,train_y)
pred_test_linear=model_linear.predict(test_x)
np.mean(pred_test_linear==test_y) #accuracy=99.72%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_x,train_y)
pred_test_poly = model_poly.predict(test_x)
np.mean(pred_test_poly==test_y) # Accuracy = 100%


#kernel=rbf
model_rbf=SVC(kernel="rbf")
model_rbf.fit(train_x,train_y)
pred_test_rbf=model_rbf.predict(test_x)
np.mean(pred_test_rbf==test_y) #accuracy=100%

# kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_x,train_y)
pred_test_sig = model_sig.predict(test_x)

np.mean(pred_test_sig==test_y) # Accuracy = 73.40


