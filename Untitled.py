#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import pandas as pd
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
from sklearn.metrics import accuracy_score

gammas = [0.01, 0.005, 0.001]
C_list =  [0.1, 0.2, 0.5, 0.7, 1, 2, 5]
train_frac = 0.1
test_frac = 0.1
dev_frac = 0.1

X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=1-train_frac, shuffle=True
)

X_test,X_dev, y_test,y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/(1-train_frac), shuffle=True
)


# In[72]:


df =pd.DataFrame()

pg = []
pc =[]
acc =[]

for gamma in gammas:
    for c in C_list:
        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=gamma, C=c)
        
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the dev subset
        predicted_dev = clf.predict(X_dev)
        
        score = accuracy_score(y_pred=predicted_dev,y_true=y_dev)
        
        pg.append(gamma)
        pc.append(c)
        acc.append(score)
        
predicted = clf.predict(X_test)
        
        
df['train'] = pg
df['dev']= pc
df['Accuracy'] = acc

df

