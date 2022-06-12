#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Practical use case of Classification ML for Students Campus Placement


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[5]:


df=pd.read_csv('MBA_Placement_Data.csv')
df.shape


# In[6]:


df.head


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


mean_salary=df['salary'].mean()
mean_salary


# In[13]:


df['salary']=df['salary'].fillna=mean_salary


# In[ ]:





# In[19]:


#X=df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
#df.replace ({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
#df.replace({'Transmission':{'Automatic':0,'Manual':1}},inplace=True)
#X=df.drop(['Car_Name','Selling_Price'],axis=1)


# In[30]:


#Use pd.dummy function to replace text values with numerical Values
numerical_values=pd.get_dummies(df[['gender','ssc_b','hsc_b','hsc_s','degree_t','specialisation']])
numerical_values


# In[15]:


df=pd.concat([df,numerical_values],axis=1)


# In[16]:


X=df.drop(['status','gender','hsc_b','ssc_b','hsc_b','hsc_s','degree_t','specialisation','workex' ],axis=1)


# In[62]:





# In[17]:


y=df['status']
#y=df.replace({'status':{'Placed':1,'Not Placed':0}},inplace=True)
print(y)


# In[ ]:





# In[18]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=62)


# In[81]:


X_test


# In[31]:


# Now apply the Classification Fitment Function from LinearRegression


# In[19]:


lr=LogisticRegression(solver='liblinear')


# In[20]:


lr.fit(X_train,y_train)


# In[ ]:


# Apply the testing Data to Predict Results


# In[21]:


pred=lr.predict(X_test)


# In[60]:


# Compare the Actual Test Data with Predicted values


# In[22]:


metrics.accuracy_score(y_test,pred)


# In[55]:


#Change Testing to Training Percentage Split in line 74 and check the results


# In[25]:





# In[26]:





# In[ ]:





# In[23]:


sb.countplot(df['status'],hue=df['specialisation'])


# In[33]:


#sb.barplot(x=df['degree_t'], y=df['status'])
#we have to use the Dummy values of degree_t and status


# In[ ]:




