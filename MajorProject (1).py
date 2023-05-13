#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data

import pandas as pd

df = pd.read_csv('/Users/PC/Downloads/Housing.csv')
df


# In[2]:


df.shape


# In[3]:


df.size


# In[4]:


df.info()


# In[67]:


a = df.iloc[0:20,0:2]
a


# In[68]:


a.area.nunique()


# In[69]:


a.area.value_counts()


# In[70]:


a['area'].unique()


# In[71]:


a.area.value_counts()


# In[73]:


#data visualization
import pandas as pd
import matplotlib.pyplot as plt
#plt.scatter(x-axis,y-axis)
plt.scatter(a['area'],a['price'])
plt.title('area vs Prices')
plt.xlabel('area')
plt.ylabel('Prices')


# In[74]:


x = a.iloc[:,0:1].values
x


# In[75]:


y = a.iloc[:,1].values
y


# In[76]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[77]:


model.fit(x,y)


# In[78]:


y_pred = model.predict(x) #using the input values, we predict the output
y_pred


# In[79]:


y


# In[81]:


model.predict([[7000]])


# In[82]:


m = model.coef_ # to find value of m
print(m)

c = model.intercept_
print(c)


# In[83]:


#Visualisation for the best fit line

plt.scatter(x,y)
plt.plot(x,y_pred,color = 'Orange')
plt.title('Best Fit Line')
plt.xlabel('Area')
plt.ylabel('Prices')


# In[ ]:




