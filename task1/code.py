#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold


# In[25]:


# Import data into dataframe

df = pd.read_csv('train.csv')
df.head()


# In[35]:


# Transform dataframe into arrays of testpoints x and labels y

y = np.array(df['y'])
x = (df.to_numpy())[:,1:]
x, y


# In[26]:


# K-fold split for cross validation

kf = KFold(n_splits=10)
kf


# In[32]:


# Verify proper splitting of test set into 10 parts

for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)


# In[40]:


# Define 5 different ridge models with different value for lambda

lambdas = [0.1, 1, 10, 100, 200]
models = []
for l in lambdas:
    models.append(linear_model.Ridge(alpha = l))
models


# In[47]:


# Do k-fold ridge regression with the splits and models previously defined. Take the mean of the resulting RMSE
# and store that in a list

import math
import statistics

results = []
for clf in models:
    temp = []
    for train_index, test_index in kf.split(x):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print("MODEL:", clf)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        mse = metrics.mean_squared_error(y_test, pred)
        rmse = math.sqrt(mse)
        # print("RMSE:", rmse)

        temp.append(rmse)
    results.append(statistics.mean(temp))
results


# In[46]:


# Write final results into a CSV

out = pd.DataFrame(results)
out.to_csv("submission.csv", header=False, index=False)


# In[ ]:




