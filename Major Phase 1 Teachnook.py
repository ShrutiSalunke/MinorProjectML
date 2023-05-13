#!/usr/bin/env python
# coding: utf-8

# In[2]:


#data

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/areavsprices.csv')
df


# In[3]:


#preprossesing 
df.shape


# In[4]:


df.size


# In[5]:


df.info()


# In[6]:


#data visualization

import matplotlib.pyplot as plt
#plt.scatter(x-axis,y-axis)
plt.scatter(df['Area'],df['Prices'])
plt.title('Area vs Prices')
plt.xlabel('Area')
plt.ylabel('Prices')


# In[7]:


#Input - Area
#Output - Prices


# In[17]:


#4. Divide the data into input and output

#input(x) is always 2d array
#output(y) is always 1d array

x = df.iloc[:,0:1].values
x


# In[19]:


y = df.iloc[:,1].values
y


# In[ ]:


#5. train and test variables
# needs more data in this case


# In[12]:


#6. Normalisation or scaling (done only for inputs) and done for multivariate datasets
#As our dataset is univariate , we are not performing this step


# In[13]:


#Run a classifier, Regressor or Clusterer

from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[20]:


#8. Fit the model(Mapping/Plotting the inputs with the outputs)
#LinearRegression(x,y)
model.fit(x,y) #We are plotting the x and y values in the LinearRegression Library


# In[21]:


#9. Predict the output
y_pred = model.predict(x) #using the input values, we predict the output
y_pred #predicted output values


# In[22]:


y #Actual output values to compare


# In[23]:


#Conclusion :we have to compare y_pred and y (predicted and actual values)
#when we compare , we come tyo know there is a huge diff
#so this huge diff does not mean that model is wrong
#It only means that our model is not linear/less linear
# Linearity of a model depends on nature of data as well as size


# In[24]:


#individual prediction

model.predict([[2000]])#price for 2000 sq ft


# In[25]:


#cross verification technique
# y = mx + c ...Equation of a straight line
# m = Slope
# c - Constant/ Y-intercept
# y - dependant variable
# x - independant variable


# In[26]:


# y = mx + c

m = model.coef_ # to find value of m
print(m)

c = model.intercept_
print(c)


# In[28]:


# substitute into the formula
#y = mx + c
m*2000 + c


# In[31]:


#Visualisation for the best fit line

plt.scatter(x,y)
plt.plot(x,y_pred,color = 'Orange')
plt.title('Best Fit Line')
plt.xlabel('Area')
plt.ylabel('Prices')


# In[114]:


#data

import pandas as pd

df = pd.read_csv('/Users/PC/Desktop/FastFoodRestaurants.csv')
df


# In[115]:


df.shape


# In[116]:


df.size


# In[117]:


df.info()


# In[118]:


a = df[0:51]
a


# In[119]:


a.shape


# In[120]:


a.size


# In[121]:


a.info()


# In[122]:


a.iloc[0:21,0:2]


# In[123]:


a.city.nunique()


# In[124]:


a['city'].nunique()


# In[125]:


a.city.value_counts()


# In[126]:


a['city'].unique()


# In[127]:


a.city.value_counts()


# In[128]:


a.groupby('city').size()


# In[ ]:





# In[ ]:





# In[ ]:




