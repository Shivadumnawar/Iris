# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:34:02 2020

@author: shiva dumnawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
iris= load_iris()

dir(iris)

df= pd.DataFrame(iris.data, columns= iris.feature_names)
df.head()

df['target']= iris.target
df.head()

print(iris.target_names)

df['target']= df['target'].apply(lambda x: iris.target_names[x])

df.info()

df_des= df.describe()

# check null values
df.isnull().sum()
''' no null values '''

# check outliers
df.plot(kind='box')
''' There are outliers in sepal width (cm) feature '''

df['target'].value_counts()

plt.figure(figsize=(8,6))
sns.boxplot(x= 'target', y='sepal length (cm)', data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'target', y='sepal width (cm)', data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'target', y='petal length (cm)', data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'target', y='petal width (cm)', data= df)

plt.figure(figsize=(12,8))
sns.scatterplot(x= 'sepal length (cm)', y= 'sepal width (cm)', hue='target', data= df)

plt.figure(figsize=(12,8))
sns.scatterplot(x= 'petal length (cm)', y= 'petal width (cm)', hue='target', data= df)

# label encoder
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

df['target']= le.fit_transform(df['target'])

# correlation
plt.figure(figsize=(12,8))
c= df.corr()
sns.heatmap(c, cmap= 'coolwarm', annot= True, linewidth=0.5)
plt.yticks(rotation=0)

X= df.iloc[:, :-1]
y= df['target'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

# scaling
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

X_train= ss.fit_transform(X_train)
X_test= ss.fit_transform(X_test)

from sklearn.svm import SVC
# svm with linear kernel
clf= SVC(kernel='linear')

clf.fit(X_train, y_train.ravel())

pred= clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print(accuracy_score(y_test, pred))

print(classification_report(y_test, pred))


# svm with rbf kernel
model= SVC()

model.fit(X_train, y_train.ravel())

y_pred= model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


""" In this problem, svm with linear kernel is giving more accuracy than rbf kernel"""