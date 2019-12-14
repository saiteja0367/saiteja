import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sb
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
filename ='C:/Users/sanjay/Downloads/customersdata.json'

with open(filename,'r')  as inputFile:      input = json.loads("[" + 
        inputFile.read().replace("}\n{", "},\n{") + 
    "]")

#type(input)

#print(input["orders"])
#a=0

df = pd.DataFrame( columns = ['customerDevice','customerEmail','customerIPAddress','customerPhone','customerBillingAddress',
                              'fraudulent','orderAmount','orderId','orderShippingAddress','orderState']) 

#input.pop(167)
a=0 
for i in input: 
   # print(type(i))
   # print(i["fraudulent"])
    
    for j in (i["orders"]):
        #print(type(j))
       # print(j["orderId"])
        df.loc[a] = (i["customer"]["customerDevice"],i["customer"]["customerEmail"],i["customer"]["customerIPAddress"],i["customer"]["customerPhone"],i["customer"]["customerBillingAddress"],i["fraudulent"],j["orderAmount"],j["orderId"],j["orderShippingAddress"],j["orderState"])
        a=a+1

a=0 
df_t = pd.DataFrame( columns = ['orderId','paymentMethodId','transactionAmount','transactionFailed','transactionId'])
    
for i in input: 
      for j in (i["transactions"]):
       df_t.loc[a]=(j["orderId"],j["paymentMethodId"],j["transactionAmount"],j["transactionFailed"],j["transactionId"])
       a=a+1

df_p = pd.DataFrame( columns = ['paymentMethodId','paymentMethodIssuer','paymentMethodProvider','paymentMethodRegistrationFailure','paymentMethodType',])
a=0 

 for i in input: 
      for j in (i["paymentMethods"]):
       df_p.loc[a]=(j["paymentMethodId"],j["paymentMethodIssuer"],j["paymentMethodProvider"],j["paymentMethodRegistrationFailure"],j["paymentMethodType"])
       a=a+1

df_transaction = df.merge(df_t, how='left')
df_final = df_transaction.merge(df_p, how='left')
df_final.to_csv("customers.csv",encoding="utf-8")

import os

os.getcwd()

df_final.columns
df_final.info     ###623x18###
df_final.fraudulent.value_counts()
###False    366
###True     257
###duplicates###
dupes=df_final.duplicated()
sum(dupes)
###labelencoder
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df_final['customerEmail']=lb.fit_transform(df_final['customerEmail'])
df_final['customerBillingAddress']=lb.fit_transform(df_final['customerBillingAddress'])
df_final['orderState']=lb.fit_transform(df_final['orderState'])
df_final['transactionFailed']=lb.fit_transform(df_final['transactionFailed'])
df_final['paymentMethodProvider']=lb.fit_transform(df_final['paymentMethodProvider'])
df_final['paymentMethodIssuer']=lb.fit_transform(df_final['paymentMethodIssuer'])
df_final['paymentMethodType']=lb.fit_transform(df_final['paymentMethodType'])
df_final['paymentMethodRegistrationFailure']=lb.fit_transform(df_final['paymentMethodRegistrationFailure'])
df_final['fraudulent']=lb.fit_transform(df_final['fraudulent'])
x=df_final.drop(["fraudulent"],axis=1)
y=df_final["fraudulent"]
y['fraudulent']=lb.fit_transform(y['fraudulent'])

###feature selection###
from feature_selector import FeatureSelector
from feature_selector import FeatureSelector

from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
sd=f_classif(x,y)
sklearn.feature_selection.f_classif(x,y)

mut_info_score = mutual_info_classif(x,y)

x.drop(['customerDevice','customerIPAddress','customerPhone','orderId','orderShippingAddress','paymentMethodId','transactionId'],inplace=True,axis=1)
####Future Selection####
from sklearn.tree import  ExtraTreeClassifier

model = ExtraTreeClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(19).plot(kind='barh')
###In feature selection we found that none of the variables are insiginifcant####


########################### EDA ########################

#Identify duplicates records in the data
######

des=x.describe() #mean,Std. dev, range
y.describe()
claim1.median()
y=claim1.mode()
x.var()
y.var()
x.skew()
#customerEmail                       0.085141
customerBillingAddress             -0.042003
orderAmount                         6.320424
orderState                         -0.210712
transactionAmount                   6.320424
transactionFailed                   1.040565
paymentMethodIssuer                 0.637297
paymentMethodProvider              -0.107239
paymentMethodRegistrationFailure    2.184514
paymentMethodType                  -1.418318

y.skew()
####0.35625995529952165

#histograms for each variable in df
hist =x.hist(bins=20,figsize =(14,14)) #data are not normally distributed#

sns.countplot(data = x, x = 'customerEmail')
sns.countplot(data = x, x = 'customerBillingAddress')
sns.countplot(data = x, x = 'orderAmount')
sns.countplot(data = x, x = 'orderState')
sns.countplot(data = x, x = 'transactionAmount')
sns.countplot(data = x, x = 'transactionFailed')
sns.countplot(data = x, x = 'paymentMethodIssuer')
sns.countplot(data = x, x = 'paymentMethodProvider')
sns.countplot(data = x, x = 'paymentMethodRegistrationFailure')
sns.countplot(data = x, x = 'paymentMethodType')

#create a boxplot for every column in df
boxplot = x.boxplot(grid=True, vert=True,fontsize=13)

#create the correlation matrix heat map
plt.figure(figsize=(14,12))
sns.heatmap(x.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0)

#pair plots
 sns.pairplot(x)

x.isnull().sum() #no null values

#######cross validation ######


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=24)
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(classifier,x,y,cv=10)
print(all_accuracies)
print(all_accuracies.mean()) #75%

import catboost as ctb
modell = ctb.CatBoostClassifier()
all_accuraciess = cross_val_score(modell,x,y,cv=10)
print(all_accuraciess)
print(all_accuraciess.mean()) #73.17%


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
all_accuraciesss = cross_val_score(abc,x,y,cv=10)
print(all_accuraciesss)
print(all_accuraciesss.mean()) #71.56186%

from xgboost import XGBClassifier
model = XGBClassifier()
all_accuracy = cross_val_score(model,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #64.31%

from sklearn.ensemble import GradientBoostingClassifier
model2=GradientBoostingClassifier()
all_accuracy=cross_val_score(model2,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #75.25%

from sklearn.tree import DecisionTreeClassifier
model3=DecisionTreeClassifier()
all_accuracy=cross_val_score(model3,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #68.87%

from sklearn.neural_network import MLPClassifier
model4=MLPClassifier()
all_accuracy=cross_val_score(model4,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #66.93%


from sklearn.linear_model import LogisticRegression
model5=LogisticRegression()
all_accuracy=cross_val_score(model5,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #65.63%

from sklearn.linear_model import SGDClassifier
model6=SGDClassifier()
all_accuracy=cross_val_score(model6,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #55.02%

from sklearn.neighbors import KNeighborsClassifier as KNC
model7=KNC()
all_accuracy=cross_val_score(model7,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #62.47%

from sklearn.linear_model import RidgeClassifier
model8=RidgeClassifier()
all_accuracy=cross_val_score(model8,x,y,cv=10)
print(all_accuracy)
print(all_accuracy.mean()) #64.5%


#######from above models we can infer that Randomforest classifier has got more accuracy so we will build our model with RandomForest classifier######

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from catboost import CatBoostClassifier
rnd_state=42
final_model=CatBoostClassifier(random_seed=rnd_state,custom_metric='Accuracy')
final_model.fit(x_train,y_train)
final_model.score(x_test,y_test) #96%

########random forest classification######

from sklearn.ensemble import RandomForestClassifier as RF
final_model=RF(n_estimators=10,criterion='entropy',max_depth=5,n_jobs=4,oob_score=True)

final_model.fit(x_train,y_train)
pred=final_model.predict(x_train)
pred1=final_model.predict(x_test)
pd.crosstab(y_train,pred,rownames=["ACTUAL"],colnames=["PREDICTORS"])
pd.crosstab(y_test,pred1,rownames=["ACTUAL"],colnames=["PREDICTORS"])
np.mean(pred==y_train)##82%
np.mean(pred1==y_test)##81%

#######neural networks#######


from keras.utils import to_categorical
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes) #2


model_nn = Sequential()
model_nn.add(Dense(500, activation='relu', input_dim=10))
model_nn.add(Dense(100, activation='relu'))
model_nn.add(Dense(50, activation='relu'))
model_nn.add(Dense(2, activation='softmax'))

# Compile the model
model_nn.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


model_nn.fit(x_train, y_train, epochs=20)
pred_train= model_nn(x_train)
scores = model_nn.evaluate(x_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 ####Accuracy on training data: 0.5883533954620361% 
 ###### Error on training data: 0.41164660453796387
pred_test= model_nn.predict(x_test)
scores2 = model_nn.evaluate(x_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    
  
##### Accuracy on test data: 0.5839999914169312% 
##### Error on test data: 0.41600000858306885

#inferences:By these we can conclude that Random Forest classification gives best accuracy compared to other models












