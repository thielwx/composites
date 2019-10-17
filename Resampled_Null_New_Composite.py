#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4 as nc
import numpy as np

from scipy import interpolate
import os
from datetime import datetime
import pandas as pd
from pandas import HDFStore
import csv
import glob
import datetime as DT
import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

from pyproj import Proj
from pyresample import image, geometry, SwathDefinition


# In[ ]:


args = sys.argv

ABIp = args[1]
GLMp = args[2]
case = args[3]


# In[ ]:


#Dictionary to make plotting easier and more consistent
#   0=File name (ABI only)
#   1=netCDF variable name
#   2=Title
#   3=Y-axis

v = {
    'CMIP13':['CMIPC13','CMI'                  ,'ABI Cloud Top Brightness Temperature - Band 13','Brightness Temperature (K)'],
    'ACHA':  ['ACHAC'  ,'HT'                   ,'ABI Cloud Top Height'                          ,'Cloud Top Height (m)'      ],
    'CTP':   ['CTPC'   ,'PRES'                 ,'ABI Cloud Top Pressure'                        ,'Cloud Top Pressure (hPa)'  ],
    'ACTP':  ['ACTPC'  ,'Phase'                ,'ABI Cloud Top Phase'                           ,'Phase'                     ],
    'FED':   ['FED'    ,'flash_extent_density' ,'GLM Flash Extent Density'                      ,'Flash Extent Density (Flashes per 5 min.)'],
    'FE':    ['FE'     ,'flash_energy'         ,'GLM Total Flash Energy'                        ,'Energy (fJ)'               ],
    'AFA':   ['AFA'    ,'average_flash_area'   ,'GLM Average Flash Area'                        ,'Flash Area (km 2)'         ],
    'MFA':   ['MFA'    ,'minimum_flash_area'   ,'GLM Minimum Flash Area'                        ,'Flash Area (km 2)']
}


# In[4]:


#Color Formatting

sns.set(color_codes=True)
sns.set(style='ticks')
colors = ['dodger blue', 'sea green', 'green', 'lime', 'yellow', 'orange']
xkcd = sns.xkcd_palette(colors)
sns.set_palette(xkcd)
#sns.palplot(sns.xkcd_palette(colors))


# In[37]:


#Data Locations
ABI_loc = '/localdata/cases/'+case+'/ABI/'+v[ABIp][0]+'/'
GLM_loc = '/localdata/cases/'+case+'/GLM5/'
save_pic_loc = '/localdata/cases/'+case+'/resampled_pics/'
save_file_loc = '/localdata/cases/'+case+'/resampled_data/'

ABI_filelist = sorted(os.listdir(ABI_loc))
halfway = len(ABI_filelist) / 2


# In[38]:


#Setting up the resampled swaths
lat2 = np.load('/localdata/coordinates/2km_lat_grid.npy')
lon2 = np.load('/localdata/coordinates/2km_lon_grid.npy')
lat10= np.load('/localdata/coordinates/10km_lat.npy')
lon10= np.load('/localdata/coordinates/10km_lon.npy')
swath_def10 = SwathDefinition(lons=lon10,lats=lat10)
swath_def2 = SwathDefinition(lons=lon2,lats=lat2)


# ## Step 1: Generating the composite datasets

# In[39]:


GLM_composite_a = np.zeros((1,150,250)).astype(np.float16)
GLM_composite_b = np.zeros((1,150,250)).astype(np.float16)
ABI_composite_a = np.zeros((1,150,250)).astype(np.float16)
ABI_composite_b = np.zeros((1,150,250)).astype(np.float16)


for file in sorted(os.listdir(ABI_loc)):
    #Loading in the ABI data
    ABI_data = nc.Dataset(ABI_loc+file, 'r')
    ABI_x = ABI_data.variables['x'][:] 
    ABI_y = ABI_data.variables['y'][:]
    ABI_var = ABI_data.variables[v[ABIp][1]][:,:]
    ABI_var = np.ma.filled(ABI_var,fill_value=0)
    
    #ABI time stuff
    ABI_time = datetime.strptime(ABI_data.time_coverage_end, '%Y-%m-%dT%H:%M:%S.%fZ')
    print (ABI_time)
    YYYY = ABI_time.strftime('%Y')
    mm = ABI_time.strftime('%m') #month
    dd = ABI_time.strftime('%d')
    HH = ABI_time.strftime('%H')
    MM = ABI_time.strftime('%M') #Minute
    
    #Loading in the GLM data
    GLM_file = 'GLM5-'+YYYY+mm+dd+'-'+HH+MM+'.nc'
    GLM_data = nc.Dataset(GLM_loc+GLM_file,'r')
    GLM_x = np.load('/localdata/coordinates/ABI/x2km.npy') 
    GLM_y = np.load('/localdata/coordinates/ABI/y2km.npy')
    GLM_var = GLM_data.variables[v[GLMp][1]][:,:]
    GLM_var = np.ma.filled(GLM_var,fill_value=0)
    
    if (ABIp == 'CMIP13') | (ABIp == 'ACTP'):
        ABI_new = ABI_var
    else:
        swath_con = image.ImageContainerNearest(ABI_var,swath_def10,radius_of_influence=10000)
        swath_resampled = swath_con.resample(swath_def2)
        ABI_new = swath_resampled.image_data
    
    #Creating the composites
    ABI_new = np.expand_dims(ABI_new,axis=0)
    GLM_var = np.expand_dims(GLM_var,axis=0)
    
    
    if GLM_composite_a.shape[0] < halfway:
        ABI_composite_a = np.append(ABI_composite_a,ABI_new[:,::10,::10],axis=0)
        GLM_composite_a = np.append(GLM_composite_a,GLM_var[:,::10,::10],axis=0)
    else:
        ABI_composite_b = np.append(ABI_composite_b,ABI_new[:,::10,::10],axis=0)
        GLM_composite_b = np.append(GLM_composite_b,GLM_var[:,::10,::10],axis=0)
    
    ABI_data.close()
    GLM_data.close()
    
ABI_composite = np.append(ABI_composite_a,ABI_composite_b,axis=0)
GLM_composite = np.append(GLM_composite_a,GLM_composite_b,axis=0)


# ## Step 2: Check for overlapping ABI/GLM points

# In[40]:


GLM_overlap = GLM_composite.astype(int)
ABI_overlap = ABI_composite.astype(int)

#Boolean array where GLM data are nonzero
GLM_composite_loc = GLM_composite > 0

#Applying from one dataset to the other
ABI_overlap[GLM_composite_loc] = 0.0

#overlap is the main dataset now
print ('Overlap check complete')


# In[41]:


del (GLM_composite,ABI_composite)


# ## 3. Isolating sections by viewing angle

# In[42]:


angle = np.load('/localdata/coordinates/GLM/viewing_angle2.npy')[::10,::10]
#Setting up boolean arrays to extract angles
f = (angle>=70.0) & (angle<80.0)
e = (angle>=60.0) & (angle<70.0)
d = (angle>=50.0) & (angle<60.0)
c = (angle>=40.0) & (angle<50.0)
b = (angle>=30.0) & (angle<40.0)
a = (angle>=20.0) & (angle<30.0)

del(angle)

a = np.broadcast_to(a,GLM_overlap.shape)
b = np.broadcast_to(b,GLM_overlap.shape)
c = np.broadcast_to(c,GLM_overlap.shape)
d = np.broadcast_to(d,GLM_overlap.shape)
e = np.broadcast_to(e,GLM_overlap.shape)
f = np.broadcast_to(f,GLM_overlap.shape)

ABI_a = ABI_overlap[a]
ABI_b = ABI_overlap[b]
ABI_c = ABI_overlap[c]
ABI_d = ABI_overlap[d]
ABI_e = ABI_overlap[e]
ABI_f = ABI_overlap[f]

del(a,b,c,d,e,f)

ABI_a_num = ABI_a[np.nonzero(ABI_a)]
ABI_b_num = ABI_b[np.nonzero(ABI_b)]
ABI_c_num = ABI_c[np.nonzero(ABI_c)]
ABI_d_num = ABI_d[np.nonzero(ABI_d)]
ABI_e_num = ABI_e[np.nonzero(ABI_e)]
ABI_f_num = ABI_f[np.nonzero(ABI_f)]

del (ABI_a,ABI_b,ABI_c,ABI_d,ABI_e,ABI_f)

print ('Binning complete')


aa = len(ABI_a_num)
bb = len(ABI_b_num)
cc = len(ABI_c_num)
dd = len(ABI_d_num)
ee = len(ABI_e_num)
ff = len(ABI_f_num)

i = np.max([aa,bb,cc,dd,ee,ff])

Data_20_ABI = np.concatenate((np.ones(i-aa)*np.nan,ABI_a_num))
Data_30_ABI = np.concatenate((np.ones(i-bb)*np.nan,ABI_b_num))
Data_40_ABI = np.concatenate((np.ones(i-cc)*np.nan,ABI_c_num))
Data_50_ABI = np.concatenate((np.ones(i-dd)*np.nan,ABI_d_num))
Data_60_ABI = np.concatenate((np.ones(i-ee)*np.nan,ABI_e_num))
Data_70_ABI = np.concatenate((np.ones(i-ff)*np.nan,ABI_f_num))

del (ABI_a_num,ABI_b_num,ABI_c_num,ABI_d_num,ABI_e_num,ABI_f_num)

Data_ABI = np.stack((Data_20_ABI,Data_30_ABI,Data_40_ABI,Data_50_ABI,Data_60_ABI,Data_70_ABI),axis=1)
ABI_pd = pd.DataFrame(Data_ABI, columns=['20-30 '+str(aa),'30-40 '+str(bb),'40-50 '+str(cc),'50-60 '+str(dd),'60-70 '+str(ee),'70-80 '+str(ff)])

del(Data_ABI)

print ('DataFrame created')


# In[43]:


#Reloading the pickled files
#ABI_pd2 = pd.read_pickle('/localdata/cases/test_case/null_data/ACTP_binned_null.pkl')


# In[44]:


#Saving the data as a pickle (?) files, that way we can reload and display them faster in the future
ABI_pd.to_pickle(save_file_loc+ABIp+'_null_5min.pkl')
print ('DataFrame saved')


# In[45]:


if (ABIp == 'CMIP13') | (ABIp == 'CTP') | (ABIp == 'ACHA'):
    fig1 = plt.figure(figsize=(16, 8))
    fig1 = sns.violinplot(data=ABI_pd,cut=0)

    if (ABIp == 'CMIP13') | (ABIp == 'CTP'):
        fig1.invert_yaxis() #Only for CTP and CMIP13

    plt.title('Null '+v[ABIp][2]+' ('+ABIp+':5 min), '+case+' 12Z-12Z')
    plt.ylabel(v[ABIp][3])
    plt.grid(True)
    
    plt.savefig(save_pic_loc+ABIp+'_null_5min.png')


# In[28]:


if ABIp == 'ACTP':
    bins = np.arange(0.5,6.5,1)
    ABI_pd.hist(bins=bins, figsize=(15,15))
    plt.savefig(save_pic_loc+'Null '+ABIp+'_flash_5min.png')


# In[ ]:




