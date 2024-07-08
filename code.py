#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score


# In[6]:


import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(filename))


# In[7]:


train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")
print(len(train_df), len(test_df))


# In[8]:


train_df.head()


# In[9]:


train_df.info()


# In[10]:


print("Checking Duplicate values:",train_df.duplicated().sum(), test_df.duplicated().sum())
print("Checking Null values:", train_df.isna().sum().sum(), test_df.isna().sum().sum())


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# data preprocessing for Training Data
train_data = train_df[['gender', "category", "amt", "city", "job", "is_fraud"]]

train_data.loc[:, 'category'] = le.fit_transform(train_data["category"])
train_data.loc[:, 'city'] = le.fit_transform(train_data["city"])
train_data.loc[:, 'job'] = le.fit_transform(train_data["job"])
train_data.loc[:, 'gender'] = train_data["gender"].map({'M': 0, 'F': 1})

# data preprocessing for Testing Data
test_data = test_df[['gender',"category","amt","city", "job","is_fraud"]]

test_data.loc[:, 'category'] = le.fit_transform(test_data["category"])
test_data.loc[:, 'city'] = le.fit_transform(test_data["city"])
test_data.loc[:, 'job'] = le.fit_transform(test_data["job"])
test_data.loc[:, 'gender'] = test_data["gender"].map({'M': 0, 'F': 1})


# In[12]:


train_data.head(10), test_data.head(10)


# In[13]:


X_train = train_data[['gender', "category", "amt", "city", "job"]]
Y_train = train_data["is_fraud"]

X_test = test_data[['gender', "category", "amt", "city", "job"]]
Y_test = test_data["is_fraud"]


# In[14]:


lr_model = LogisticRegression(penalty="l1", solver="liblinear")
lr_model.fit(X_train, Y_train)


# In[15]:


Y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Testing Accuracy Score:", accuracy)


# In[16]:


from sklearn.naive_bayes import GaussianNB

gnb_model = GaussianNB()
gnb_model.fit(X_train, Y_train)
Y_pred_gnb = gnb_model.predict(X_test)
gnb_accuracy = accuracy_score(Y_test, Y_pred_gnb)
print("Training Accuracy Score:", gnb_accuracy)


# In[ ]:




