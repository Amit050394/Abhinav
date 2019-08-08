#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


train1=pd.read_csv("C:/Users/hp/Downloads/citibank-defaulters2/train_1.csv")


# In[6]:


train2=pd.read_csv("C:/Users/hp/Downloads/citibank-defaulters2/train_2.csv")


# In[7]:


train3=pd.read_csv("C:/Users/hp/Downloads/citibank-defaulters2/train_3.csv")


# In[8]:


test=pd.read_csv("C:/Users/hp/Downloads/citibank-defaulters2/test.csv")


# In[12]:


train = pd.concat([train1,train2,train3], axis=1)


# In[14]:


train.shape


# In[30]:


train.head()


# In[36]:


new_train=train.drop("Ref.No",axis=1)


# In[38]:


new_train.head()


# In[39]:


new_train.shape


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)


# In[41]:


x = new_train.drop("Y",axis=1)
y = new_train["Y"]


# In[44]:


x.shape,y.shape


# In[52]:


test.head()


# In[54]:


new_test=test.drop(["Ref.No","Y"],axis=1)


# In[55]:


new_test.head()


# In[58]:


new_test.shape


# In[61]:


x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=123,test_size=0.3)


# In[68]:


new_train.shape


# In[70]:


new_test.shape


# In[71]:


X = new_train.drop('Y', axis = 1)
y = new_train['Y']


# In[73]:


sc = StandardScaler()
x_train = sc.fit_transform(x)
x_test = sc.transform(new_test)


# In[75]:


x_train.shape


# In[76]:


x_test.shape


# In[77]:


tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
tree.fit(x_train, y)
y_pred = tree.predict(x_test)


# # Bagging

# In[94]:


submit = pd.DataFrame({'Ref.No':np.arange(1,2400),'Y':y_pred})


# In[97]:


submit.to_csv('Decision Tree Prediction.csv', index = False)


# In[82]:


from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# In[86]:


bagg = BaggingClassifier(base_estimator=tree, n_estimators = 100)
bagg.fit(x_train, y)
baggpred = bagg.predict(x_test)


# In[90]:


submit = pd.DataFrame({'Ref.No':np.arange(1,2400), 'Y':baggpred})


# In[92]:


submit.to_csv("Bagging.csv", index = False)


# # adaboost

# In[100]:


ada = AdaBoostClassifier()
ada.fit(x_train,y)
adapred = ada.predict(x_test)


# In[101]:


submit = pd.DataFrame({'Ref.No':np.arange(1,2400), 'Y':adapred})


# In[102]:


submit.to_csv("AdaBoost.csv", index = False)


# # gradint

# In[103]:


gbm = GradientBoostingClassifier()
gbm.fit(x_train,y)
gbmpred = gbm.predict(x_test)


# In[104]:


submit = pd.DataFrame({'Ref.No':np.arange(1,2400), 'Y':gbmpred})


# In[105]:


submit.to_csv("GradientBoost.csv", index = False)


# In[ ]:




