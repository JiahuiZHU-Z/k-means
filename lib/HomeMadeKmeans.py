
# coding: utf-8

# In[258]:


import keras
import numpy as np
import pandas as pd
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt


# In[262]:


def get_distance(point_data,points_K):
    dist=[]
    for i in points_K:
        dist.append(math.sqrt(sum([(a-b)**2 for a,b in zip(point_data,i)])))
    return dist


# In[263]:


def random_points(k,df):
    points = []
    for z in range(k):
        point = []
        for i in range(df.shape[1]-1):
            point.append(np.random.randint(df.iloc[:,i].min(),df.iloc[:,i].max()))
        points.append(point)
    return points


# In[264]:


def get_new_pointsK(df,k):
    points_k =[]
    for i in range(K): # nb point = K
        df_cluster = (df.loc[df['cluster'] == i])['point_data']

        size = len(df_cluster)
        nb = df.shape[1]-3
        liste_var = [0]* nb
        point = []

        for j in range(nb):
            for h in df_cluster:
                liste_var[j] += h[j]
        point = ([x/df_cluster.shape[0] for x in liste_var])
        print(point)
        points_k.append(point)  
    return points_k  


# In[267]:


def kmeans (itr,k,df):
    
    for i in range(itr) :
        if i == 0:
            points_k = random_points(k,df)
            print(points_k)
            df['distance'] = df['point_data'].apply(lambda x: get_distance(x,points_k))
            df['cluster'] = df['distance'].apply(lambda x : np.argmin(x))
            
        else:
        
            points_k = get_new_pointsK(df,k)
            print(points_k)
            df['distance'] = df['point_data'].apply(lambda x: get_distance(x,points_k))
            df['cluster'] = df['distance'].apply(lambda x : np.argmin(x))
          
    return points_k


# In[ ]:




