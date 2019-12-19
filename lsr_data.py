#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This program is created to integrate the storm reports into the datasets


# In[135]:


import numpy as np
import pandas as pd
from scipy import spatial
from pyproj import Proj
import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import sys
import os


# ### Function Land

# In[123]:


#A function that creates an empty dataframe that will be filled with the storm reports

def blank_df(stime,etime):
    print ('Creating the empty dataframe:')
    start = datetime.strptime(stime,'%Y%m%d-%H%M')
    end = datetime.strptime(etime,'%Y%m%d-%H%M')
    #print (start,end)
    tlist = pd.date_range(start=start,end=end,freq='5T').to_pydatetime().tolist()
    
    f_array = np.full((len(xxf)),False)
    df_tot = pd.DataFrame()
    
    for i in tlist[1:]:
        print (i)
        timestr = timestring(i)
        
        tnew = np.broadcast_to(timestr,len(xxf))
        df = pd.Series(f_array,name='dummy')
        mi = pd.MultiIndex.from_arrays([tnew,xxf,yyf])
        df = df.reindex(mi)
        newdf = pd.DataFrame({'dummy':df,
                              'HAIL':f_array,
                              'WND':f_array,
                              'TOR':f_array,
                             'ANY':f_array})
        df_tot = pd.concat((df_tot,newdf),axis=0,sort=True)
    return df_tot.drop(columns='dummy'),tlist


# In[6]:


#A function that takes in lat/lon values and reprojects that data into x/y values from GOES16

def xy_convert(lat,lon):
    #Satellite constants from GOES16
    sat_h = 35786023.0
    sat_lon = -75.0
    sat_sweep = 'x'
    #Making the conversion
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    x,y = p(lon,lat)
    #Dividing by the satellite height so the values are scaled comparable to what's given in the datasets
    return x/sat_h,y/sat_h


# In[8]:


def kdtree(x,y):
    a = tree.query((x,y),k=9)[1]
    x_loc = xxf[a]
    y_loc = yyf[a]
    return x_loc,y_loc


# In[96]:


#This function takes in the LSRs and gets them ready to be inserted into the datasets

def sr_formatter(LSR,tlist,tlist_LSR):
    print ('Formatting the LSRs:')
    #Setting the index to datetime
    index = pd.to_datetime(LSR.loc[:,'VALID2'])
    LSR.loc[:,'VALID2'] = index
    LSR = LSR.set_index('VALID2')
    newdf = pd.DataFrame()
    
    
    #Looping through data five minutes at a time
    index = np.arange(0,len(tlist_LSR),1)
    for i in index[:-1]:
        #Pulling a 5 minute segment
        tLSR = LSR.between_time(tlist_LSR[i].time(),tlist_LSR[i+1].time(),include_start=False)
        #Creating the time string that will be inserted for the VALID time
        etimestr = tlist_LSR[i+1].strftime('%Y%m%d-%H%M')
        print (etimestr)
        tLSR.loc[:,'VALID'] = etimestr
        newdf = pd.concat((newdf,tLSR),axis=0)
        
    #Gathering the hail, wind, and tornado reports
    hLSR = newdf[(newdf.loc[:,'TYPETEXT']=='HAIL')]
    wLSR = newdf[((newdf.loc[:,'TYPETEXT']=='TSTM WND DMG') | (newdf.loc[:,'TYPETEXT']=='TSTM WND GST'))]
    tLSR = newdf[(newdf.loc[:,'TYPETEXT']=='TORNADO')]
    
    return hLSR, wLSR, tLSR


# In[114]:


#Expands how long the storm reports are kept in the data spatially

def grid_insert(valid,x_loc,y_loc,etype,df):
    deltas = [-15,-10,-5]
    t = datetime.strptime(valid, '%Y%m%d-%H%M')
    
    for i in deltas:
        ts = t + timedelta(minutes=i)
        
        if (ts > tlist[0]) & (ts <= tlist[-1]):
            #Creating the time string to serach in the dataset
            nt = timestring(ts)
            
            df.loc[(nt,x_loc,y_loc)][etype] = True #Setting the selected gridpoint for the type to true
            df.loc[(nt,x_loc,y_loc)]['ANY'] = True #Setting the 'ANY' column to true
            
            
    return df


# In[115]:


def timestring(time):
    y = datetime.strftime(time,'%Y')
    j = datetime.strftime(time,'%j')
    h = datetime.strftime(time,'%H')
    M = datetime.strftime(time,'%M')
    d = datetime.strftime(time,'%d')
    m = datetime.strftime(time,'%m')
    newtime = y+m+d+'-'+h+M
    
    return newtime


# In[81]:


#A function that finds the data points and replaces them
#The outer loop is done per storm report
#The inner loop is done per gridpoints near the storm report

def grid_edit(df,xLSR,etype):
    valid = xLSR.loc[:,'VALID'].values
    x,y = xy_convert(xLSR.loc[:,'LAT'].values,xLSR.loc[:,'LON'].values) #Converting the lat/lon to x/y
    n = len(x)
    
    #Outer loop per storm report
    for i in np.arange(0,n,1):
        x_loc,y_loc = kdtree(x[i],y[i]) #Getting the nearest neighbor positions from the KDtree
        #Inner loop per nearest neighbor points
        for j in np.arange(0,len(x_loc),1):
            df = grid_insert(valid[i],x_loc[j],y_loc[j],etype,df)
    
    return df


# In[92]:


def LSR_times(lstart,lend):
    start = datetime.strptime(lstart,'%Y%m%d-%H%M')
    end = datetime.strptime(lend,'%Y%m%d-%H%M')
    #print (start,end)
    tlist_lsr = pd.date_range(start=start,end=end,freq='5T').to_pydatetime().tolist()
    return tlist_lsr


# In[129]:


def timesort(start,end):
    datastart = datetime.strptime(start,'%Y%m%d')
    dataend = datetime.strptime(end,'%Y%m%d')
    delta = timedelta(minutes=30)
    
    lsrstart = datastart - delta
    lsrend = dataend + delta
    
    lsrstart = timestring(lsrstart)
    lsrend = timestring(lsrend)
    
    datastart = timestring(datastart)
    dataend = timestring(dataend)
    
    return datastart,dataend,lsrstart,lsrend


# ### Constants

# In[100]:


#Loading the x and y grids
y = np.load('/localdata/coordinates/20km_y.npy')
x = np.load('/localdata/coordinates/20km_x.npy')
xx, yy = np.meshgrid(x,y)
xxf = xx.flatten()
yyf = yy.flatten()

#Initializing the KDtree
grid = np.stack((xxf,yyf),axis=1)
tree = spatial.KDTree(grid)


# ### Work Zone

# In[137]:


#===================================================
#  Run like this: python lsr_data.py May/20190528 20190528 20190529 May
#===================================================

args = sys.argv
#args = ['May/20190528','20190528','20190529','May']

case = str(args[1])
start = str(args[2])
end = str(args[3])
name = str(args[4])
datastart,dataend,lsrstart,lsrend = timesort(start,end)

lsrfile = os.listdir('/localdata/cases/'+case+'/LSR/')

#Reading in the LSRs
LSR = pd.read_csv('/localdata/cases/'+case+'/LSR/'+lsrfile[0])


#Creates the blank dataframe that we will be editing and the list of available times
df,tlist = blank_df(datastart,dataend)
#Creates a list of times that correspond to the LSRs
tlist_lsr = LSR_times(lsrstart,lsrend)
#Formats the storm reports so they will fit within the dataframe
hLSR, wLSR, tLSR = sr_formatter(LSR,tlist,tlist_lsr)
#Finds the data and puts it in the dataframes
df = grid_edit(df,hLSR,'HAIL')
df = grid_edit(df,wLSR,'WND')
df = grid_edit(df,tLSR,'TOR')

df.to_pickle('/localdata/cases/'+name+'/all_lsr_pre15min/'+'lsr_'+start+'.pkl')

