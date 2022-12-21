#!/usr/bin/env python
# coding: utf-8

# In[1]:


#install required libraries
get_ipython().system('pip install -U kaleido')


# In[2]:


#import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import kaleido


# In[3]:


#read data which contains details about consumers who purchased an EV
data = pd.read_csv('behavioural_dataset.csv')


# In[4]:


data.describe


# In[5]:


#understand the statistics
data.describe()


# In[6]:


print(pd.isnull(data).sum())


# In[7]:


#rename columns
data.rename(columns = {'Personal loan': 'Car_loan'}, inplace = True)
data.rename(columns = {'Price' : 'EV_Price'}, inplace = True)
data.head()


# In[8]:


#Plot a graph of the car loan status with respect to marital status
sns.countplot(x = 'Marital Status', hue = 'Car_loan', data = data, palette = 'Set1')
plt.show()


# In[9]:


(data['Marital Status'].value_counts()['Married'])/((data['Marital Status'].value_counts()['Married'])+(data['Marital Status'].value_counts()['Single']))*100


# In[10]:


#Get labels and data
labels = ['Car Loan Required','Car Loan not required']
Loan_status = [data.query('Car_loan == "Yes"').Car_loan.count(),data.query('Car_loan == "No"').Car_loan.count()]

# declare exploding pie
explode = [0.1, 0]
# define Seaborn color palette to use
palette_color = sns.color_palette('pastel')
  
# plot data on chart
plt.pie(Loan_status, labels=labels, colors=palette_color, shadow = "True",
        explode=explode, autopct='%1.1f%%')
  
# display chart
plt.show()


# In[11]:


# Plot the fequency of each entry for consumer features - Age, No. 0f Dependents, Total Salary, EV_Price
plt.figure(1, figsize=(15,5))
n=0

for x in ['Age', 'No of Dependents' ,'Total Salary'  ,'EV_Price']:
  n += 1
  plt.subplot(1,4,n)
  plt.subplots_adjust(hspace=0.5, wspace=0.5)
  sns.histplot(data[x], bins= 25)
  plt.title(f'{x}')
plt.show()


# In[12]:


get_ipython().system('pip install kmodes')
from kmodes.kprototypes import KPrototypes

# Kmodes is similar to K means clustering when computing distance for continuous data using mean but for categorical data it uses the mode
# Frequency based dissimilarity measure
# Hence it is more preferrable for clustering multiple datatypes 


# In[13]:


data.head()


# In[14]:


cluster_features = list(data.columns)
cluster_data = data[cluster_features].values


# In[15]:


cluster_data[:, 0]


# In[16]:


cluster_data[:, 0] = cluster_data[:, 0].astype(float)
cluster_data[:, 4] = cluster_data[:, 4].astype(float)
cluster_data[:, 6] = cluster_data[:, 6].astype(float)
cluster_data[:, 7] = cluster_data[:, 7].astype(float)


# In[17]:


# Finding optimal number of clusters for KPrototypes

cost = []
for num_clusters in list(range(1,8)):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
    kproto.fit_predict(cluster_data, categorical=[1,2,3,5])
    cost.append(kproto.cost_)

plt.plot(cost)


# In[18]:


cost


# In[19]:


# fitting data to clusters

kproto = KPrototypes(n_clusters=2, verbose=2,max_iter=20)
clusters = kproto.fit_predict(cluster_data, categorical=[1,2,3,5])


# In[20]:


# Append the cluster data

data['Cluster'] = clusters


# In[21]:


# Average cost of the EV
data.EV_Price.mean()


# In[22]:


# Average cost of a car in segment 1 
data.EV_Price[data.Cluster==0].mean()


# In[23]:


data['EV_Price'][data.Cluster==1].max()


# In[24]:


# Average cost of a car in segment 2
data.EV_Price[data.Cluster==1].mean()


# In[25]:


data['Cluster'].value_counts(normalize=True) * 100


# In[26]:


# Segregrate each cluster

Cluster_0 = data[data.Cluster==0]
Cluster_1 = data[data.Cluster==1]


# In[27]:


data['Cluster'].value_counts()


# In[28]:


# plot the effct of salary and ev price on cluster data

plt.scatter(Cluster_0.EV_Price, Cluster_0['Total Salary'],color='red', marker = 'x', label = 'Cluster 1')
plt.scatter(Cluster_1.EV_Price, Cluster_1['Total Salary'],color='green', label = 'Cluster 2')
plt.legend(loc="upper left")

plt.xlabel('EV Price')
plt.ylabel('Total salary')
plt.show()

# there is a clear difference in segments when comparing salary and the price of EV purchased


# In[29]:


plt.scatter(Cluster_0.EV_Price, Cluster_0['Age'],color='red', marker = 'x', label = 'Cluster 1')
plt.scatter(Cluster_1.EV_Price, Cluster_1['Age'],color='green', label = 'Cluster 2')
plt.legend(loc = "upper left")

plt.xlabel('EV Price')
plt.ylabel('Age')
plt.show()


# In[30]:


from mpl_toolkits.mplot3d import Axes3D


# In[31]:


# plot influence of age 

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(Cluster_0.EV_Price, Cluster_0['Total Salary'], Cluster_0['Age'], color='red', marker = 'x', label = 'Cluster 1')
ax.scatter(Cluster_1.EV_Price, Cluster_1['Total Salary'],Cluster_1['Age'], color='green', label = 'Cluster 2')
plt.legend(loc = 'upper left')

ax.view_init(10, 20)

plt.xlabel("EV Price")
plt.ylabel("Total Salary")
ax.set_zlabel('Age')
plt.show()


# In[32]:


# plot influence of No of Dependents

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(Cluster_0.EV_Price, Cluster_0['Total Salary'], Cluster_0['No of Dependents'], color='red', marker = 'x', label = 'Cluster 1')
ax.scatter(Cluster_1.EV_Price, Cluster_1['Total Salary'],Cluster_1['No of Dependents'], color='green', label = 'Cluster 2')
plt.legend(loc = 'upper left')
ax.view_init(10,20)

plt.xlabel("EV Price")
plt.ylabel("Total Salary")
ax.set_zlabel('No of Dependents')
plt.show()
     


# In[33]:


data['No of Dependents'].value_counts()


# In[34]:


# plot the effct of no of dependents and ev price on cluster data


plt.scatter(Cluster_1.EV_Price, Cluster_1['No of Dependents'],color='green', label = 'Cluster 2')
plt.scatter(Cluster_0.EV_Price, Cluster_0['No of Dependents'],color='red', marker = 'x', label = 'Cluster 1')
plt.legend(loc="lower right")

plt.xlabel('EV Price')
plt.ylabel('No of Dependents')
plt.show()

# there is a clear difference in segments when comparing salary and the price of EV purchased


# In[ ]:




