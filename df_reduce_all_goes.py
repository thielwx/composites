#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os
import sys
import glob


# In[10]:


def blooming(data):
        
    #Removing blooming if it's there
    points = ((data['CMIP']>=270) & (data['FED']>=10) & (data['ACHA']<=4000)) | (data['CMIP']<=100)
    data = data[~points]
    
    return data


# In[11]:


#Extracting data from where the GLM and ABI data does exist

def allgoes(data,df):

    #Extracting where data from all variables exist in the same space and time
    data_locs = np.where((~np.isnan(data.iloc[:,0])) & 
                         (~np.isnan(data.iloc[:,1])) & 
                         (~np.isnan(data.iloc[:,2])) & 
                         (~np.isnan(data.iloc[:,3])) & 
                         (~np.isnan(data.iloc[:,4])) & 
                         (~np.isnan(data.iloc[:,5])))
    data_locs = np.squeeze(data_locs)
    data = data.iloc[data_locs,:]
    
    #Compiling datasets
    df = pd.concat((df,data),axis=0)
    
    return df


# In[12]:


#Extracting data from where GLM data does not exist and ABI data does exist

def noglm(data,df):
    
    data_locs = np.where((~np.isnan(data.iloc[:,0])) & 
                         (~np.isnan(data.iloc[:,1])) & 
                         (np.isnan(data.iloc[:,2])) & 
                         (np.isnan(data.iloc[:,3])) & 
                         (np.isnan(data.iloc[:,4])) & 
                         (np.isnan(data.iloc[:,5])))
    data_locs = np.squeeze(data_locs)
    data = data.iloc[data_locs,:]
    
    #Compiling datasets
    df = pd.concat((df,data),axis=0)
    
    return df


# In[13]:


#Extracting where ABI, GLM, and LSR data exist

def alldata(data,df):
    
    data_locs = np.where((~np.isnan(data.iloc[:,0])) & 
                         (~np.isnan(data.iloc[:,1])) & 
                         (~np.isnan(data.iloc[:,2])) & 
                         (~np.isnan(data.iloc[:,3])) & 
                         (~np.isnan(data.iloc[:,4])) & 
                         (~np.isnan(data.iloc[:,5])) &
                         (data.iloc[:,7]==True))
    data_locs = np.squeeze(data_locs)
    data = data.iloc[data_locs,:]
    
    #Compiling datasets
    df = pd.concat((df,data),axis=0)
    
    return df


# In[14]:


#Extracting where ABI and GLM exist, but no LSRs

def nolsr(data,df):
    
    data_locs = np.where((~np.isnan(data.iloc[:,0])) & 
                         (~np.isnan(data.iloc[:,1])) & 
                         (~np.isnan(data.iloc[:,2])) & 
                         (~np.isnan(data.iloc[:,3])) & 
                         (~np.isnan(data.iloc[:,4])) & 
                         (~np.isnan(data.iloc[:,5])) &
                         (data.iloc[:,7]==False))
    data_locs = np.squeeze(data_locs)
    data = data.iloc[data_locs,:]
    
    #Compiling datasets
    df = pd.concat((df,data),axis=0)
    
    return df


# In[15]:


args = sys.argv
#args = ['April']

case = args[1]

lsrloc = '/localdata/cases/'+case+'/all_lsr_10min/'
goesloc = '/localdata/cases/'+case+'/data/'
sloc = '/localdata/cases/'+case+'/combo_data/'

lsrlist = sorted(os.listdir(lsrloc))
goeslist = sorted(os.listdir(goesloc))
index = np.arange(0,len(lsrlist))


# In[8]:


all_goes = pd.DataFrame()


for i in index:
    print (lsrlist[i])
    lsr = pd.read_pickle(lsrloc+lsrlist[i])
    goes = pd.read_pickle(goesloc+goeslist[i])
    both = pd.concat((goes,lsr),axis=1)
    both = blooming(both)
    
    all_goes = allgoes(both,all_goes) 
    
if not os.path.exists(sloc):
    os.makedirs(sloc)

all_goes.to_pickle(sloc+case+'_all_goes4.pkl')


# In[ ]:




