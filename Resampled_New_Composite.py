#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4 as nc
import numpy as np

from scipy import interpolate
import os
from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pyproj import Proj
from pyresample import image, geometry, SwathDefinition

import sys


# In[3]:


args = sys.argv

ABIp = args[1]
GLMp = args[2]
case = args[3]


# In[2]:


#Dictionary to make plotting easier and more consistent
#   0=File name (ABI only)
#   1=netCDF variable name
#   2=Title
#   3=Y-axis

v = {
    'CMIP13':['CMIPC13','CMI'                  ,'ABI Cloud Top Brightness Temperature - Band 13','Brightness Temperature (CMIP13:K)'],
    'ACHA':  ['ACHAC'  ,'HT'                   ,'ABI Cloud Top Height'                          ,'Cloud Top Height (ACHA:m)'      ],
    'CTP':   ['CTPC'   ,'PRES'                 ,'ABI Cloud Top Pressure'                        ,'Cloud Top Pressure (hPa)'  ],
    'ACTP':  ['ACTPC'  ,'Phase'                ,'ABI Cloud Top Phase'                           ,'Phase'                     ],
    'FED':   ['FED'    ,'flash_extent_density' ,'GLM Flash Extent Density'                      ,'Flash Extent Density (FED:Flashes per 5 min.)'],
    'FE':    ['FE'     ,'total_energy'         ,'GLM Total Flash Energy'                        ,'Total Flash Energy (FE:fJ)'               ],
    'AFA':   ['AFA'    ,'average_flash_area'   ,'GLM Average Flash Area'                        ,'Average Flash Area (AFA:km 2)'         ],
    'MFA':   ['MFA'    ,'minimum_flash_area'   ,'GLM Minimum Flash Area'                        ,'Minimum Flash Area (MFA:km 2)'         ],
    'AGA':   ['AGA'    ,'average_group_area'   ,'GLM Average Group Area'                        ,'Average Group Area (AGA:km 2)'    ]
}


# In[4]:


#Color Formatting
sns.set(color_codes=True)
sns.set(style='ticks')
colors = ['dodger blue', 'sea green', 'green', 'lime', 'yellow', 'orange']
xkcd = sns.xkcd_palette(colors)
sns.set_palette(xkcd)
#sns.palplot(sns.xkcd_palette(colors))


# In[7]:


#Data Locations
ABI_loc = '/localdata/cases/'+case+'/ABI/'+v[ABIp][0]+'/'
GLM_loc = '/localdata/cases/'+case+'/GLM5/'
save_pic_loc = '/localdata/cases/'+case+'/resampled_pics/'
save_file_loc = '/localdata/cases/'+case+'/resampled_data/'

ABI_filelist = sorted(os.listdir(ABI_loc))
halfway = len(ABI_filelist) / 2


# In[8]:


#Setting up the resampled swaths
lat2 = np.load('/localdata/coordinates/2km_lat_grid.npy')
lon2 = np.load('/localdata/coordinates/2km_lon_grid.npy')
lat10= np.load('/localdata/coordinates/10km_lat.npy')
lon10= np.load('/localdata/coordinates/10km_lon.npy')
swath_def10 = SwathDefinition(lons=lon10,lats=lat10)
swath_def2 = SwathDefinition(lons=lon2,lats=lat2)


# ## Step 1: Generating the composite datasets

# In[9]:


GLM_composite_a = np.zeros((1,1500,2500)).astype(np.float16)
GLM_composite_b = np.zeros((1,1500,2500)).astype(np.float16)
ABI_composite_a = np.zeros((1,1500,2500)).astype(np.float16)
ABI_composite_b = np.zeros((1,1500,2500)).astype(np.float16)


for file in ABI_filelist:
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
        ABI_composite_a = np.append(ABI_composite_a,ABI_new,axis=0)
        GLM_composite_a = np.append(GLM_composite_a,GLM_var,axis=0)
    else:
        ABI_composite_b = np.append(ABI_composite_b,ABI_new,axis=0)
        GLM_composite_b = np.append(GLM_composite_b,GLM_var,axis=0)
    
    ABI_data.close()
    GLM_data.close()
    
ABI_composite = np.append(ABI_composite_a,ABI_composite_b,axis=0)
GLM_composite = np.append(GLM_composite_a,GLM_composite_b,axis=0)

del(ABI_composite_a,ABI_composite_b,GLM_composite_a,GLM_composite_b)


# ## Step 2: Check for overlapping ABI/GLM points

# In[10]:


#The values for the ABI_composite_loc can also be used for thresholding the ABI products

GLM_overlap = GLM_composite.copy()
ABI_overlap = ABI_composite.copy()

#Boolean array where GLM/ABI data are zero
GLM_composite_loc = GLM_composite == 0.0
ABI_composite_loc = ABI_composite == 0.0

#Applying from one dataset to the other
GLM_overlap[ABI_composite_loc] = 0.0
GLM_overlap[GLM_composite_loc] = 0.0 #we have to apply the GLM boolean for any values over zero
ABI_overlap[GLM_composite_loc] = 0.0
ABI_overlap[ABI_composite_loc] = 0.0

#overlap is the main dataset now
del(GLM_composite_loc,ABI_composite_loc)


# In[11]:


#The values for the ABI_composite_loc can also be used for thresholding the ABI products

GLM_overlap = GLM_composite.copy()
ABI_overlap = ABI_composite.copy()

#Boolean array where GLM/ABI data are zero
GLM_composite_loc = GLM_composite == 0.0
ABI_composite_loc = ABI_composite == 0.0

#Applying from one dataset to the other
GLM_overlap[ABI_composite_loc] = 0.0
GLM_overlap[GLM_composite_loc] = 0.0 #we have to apply the GLM boolean for any values over zero
ABI_overlap[GLM_composite_loc] = 0.0

#overlap is the main dataset now
del(GLM_composite_loc,ABI_composite_loc)


# ## 3. Isolating sections by viewing angle

# In[12]:


angle = np.load('/localdata/coordinates/GLM/viewing_angle2.npy')
#Setting up boolean arrays to extract angles
f = (angle>=70.0) & (angle<80.0)
e = (angle>=60.0) & (angle<70.0)
d = (angle>=50.0) & (angle<60.0)
c = (angle>=40.0) & (angle<50.0)
b = (angle>=30.0) & (angle<40.0)
a = (angle>=20.0) & (angle<30.0)

a = np.broadcast_to(a,GLM_overlap.shape)
b = np.broadcast_to(b,GLM_overlap.shape)
c = np.broadcast_to(c,GLM_overlap.shape)
d = np.broadcast_to(d,GLM_overlap.shape)
e = np.broadcast_to(e,GLM_overlap.shape)
f = np.broadcast_to(f,GLM_overlap.shape)

GLM_a = GLM_overlap[a]
GLM_b = GLM_overlap[b]
GLM_c = GLM_overlap[c]
GLM_d = GLM_overlap[d]
GLM_e = GLM_overlap[e]
GLM_f = GLM_overlap[f]

ABI_a = ABI_overlap[a]
ABI_b = ABI_overlap[b]
ABI_c = ABI_overlap[c]
ABI_d = ABI_overlap[d]
ABI_e = ABI_overlap[e]
ABI_f = ABI_overlap[f]
print ('Boolean arrays applied')

GLM_a_num = GLM_a[np.nonzero(GLM_a)]
GLM_b_num = GLM_b[np.nonzero(GLM_b)]
GLM_c_num = GLM_c[np.nonzero(GLM_c)]
GLM_d_num = GLM_d[np.nonzero(GLM_d)]
GLM_e_num = GLM_e[np.nonzero(GLM_e)]
GLM_f_num = GLM_f[np.nonzero(GLM_f)]
print ('GLM Data Extracted')
ABI_a_num = ABI_a[np.nonzero(ABI_a)]
ABI_b_num = ABI_b[np.nonzero(ABI_b)]
ABI_c_num = ABI_c[np.nonzero(ABI_c)]
ABI_d_num = ABI_d[np.nonzero(ABI_d)]
ABI_e_num = ABI_e[np.nonzero(ABI_e)]
ABI_f_num = ABI_f[np.nonzero(ABI_f)]
print ('ABI Data Extracted')

del(a,b,c,d,e,f)


# In[13]:


aa = len(GLM_a_num)
bb = len(GLM_b_num)
cc = len(GLM_c_num)
dd = len(GLM_d_num)
ee = len(GLM_e_num)
ff = len(GLM_f_num)

print (aa,bb,cc,dd,ee,ff)

i = np.max([aa,bb,cc,dd,ee,ff])

Data_20_GLM = np.concatenate((np.ones(i-aa)*np.nan,GLM_a_num))
Data_30_GLM = np.concatenate((np.ones(i-bb)*np.nan,GLM_b_num))
Data_40_GLM = np.concatenate((np.ones(i-cc)*np.nan,GLM_c_num))
Data_50_GLM = np.concatenate((np.ones(i-dd)*np.nan,GLM_d_num))
Data_60_GLM = np.concatenate((np.ones(i-ee)*np.nan,GLM_e_num))
Data_70_GLM = np.concatenate((np.ones(i-ff)*np.nan,GLM_f_num))
Data_GLM = np.stack((Data_20_GLM,Data_30_GLM,Data_40_GLM,Data_50_GLM,Data_60_GLM,Data_70_GLM),axis=1)
GLM_pd = pd.DataFrame(Data_GLM,columns=['20-30 '+str(aa),'30-40 '+str(bb),'40-50 '+str(cc),'50-60 '+str(dd),'60-70 '+str(ee),'70-80 '+str(ff)])

del (GLM_a_num,GLM_b_num,GLM_c_num,GLM_d_num,GLM_e_num,GLM_f_num)

Data_20_ABI = np.concatenate((np.ones(i-aa)*np.nan,ABI_a_num))
Data_30_ABI = np.concatenate((np.ones(i-bb)*np.nan,ABI_b_num))
Data_40_ABI = np.concatenate((np.ones(i-cc)*np.nan,ABI_c_num))
Data_50_ABI = np.concatenate((np.ones(i-dd)*np.nan,ABI_d_num))
Data_60_ABI = np.concatenate((np.ones(i-ee)*np.nan,ABI_e_num))
Data_70_ABI = np.concatenate((np.ones(i-ff)*np.nan,ABI_f_num))
Data_ABI = np.stack((Data_20_ABI,Data_30_ABI,Data_40_ABI,Data_50_ABI,Data_60_ABI,Data_70_ABI),axis=1)
ABI_pd = pd.DataFrame(Data_ABI, columns=['20-30 '+str(aa),'30-40 '+str(bb),'40-50 '+str(cc),'50-60 '+str(dd),'60-70 '+str(ee),'70-80 '+str(ff)])

if GLMp == 'FE':
    GLM_pd *= 10**6

del (ABI_a_num,ABI_b_num,ABI_c_num,ABI_d_num,ABI_e_num,ABI_f_num)


# In[ ]:


#Saving the data as a pickle (?) files, that way we can reload and display them faster in the future
if (ABIp == 'CMIP13') & (GLMp == 'FED'):
    ABI_pd.to_pickle(save_file_loc+ABIp+'_flash_5min.pkl')
    GLM_pd.to_pickle(save_file_loc+GLMp+'_5min.pkl')
elif (ABIp == 'CMIP13'):
    GLM_pd.to_pickle(save_file_loc+GLMp+'_5min.pkl')
elif (GLMp =='FED'):
    ABI_pd.to_pickle(save_file_loc+ABIp+'_flash_5min.pkl')


# In[ ]:


#Reloading the pickled files
#ABI_pd2 = pd.read_pickle(save_file_loc+'ACTP_binned.pkl')
#GLM_pd2 = pd.read_pickle(save_file_loc+'FE_5min.pkl')


# In[14]:


if ((ABIp == 'CTP') | (ABIp == 'ACHA')) | ((ABIp == 'CMIP13') & (GLMp == 'FED')):
    fig1 = plt.figure(figsize=(16, 8))
    fig1 = sns.violinplot(data=ABI_pd,cut=0)

    if (ABIp == 'CMIP13') | (ABIp == 'CTP'):
        fig1.invert_yaxis() #Only for CTP and CMIP13

    plt.title(v[ABIp][2]+' ('+ABIp+':5 min), '+case+' 12Z-12Z')
    plt.ylabel(v[ABIp][3])
    plt.grid(True)
    
    plt.savefig(save_pic_loc+ABIp+'_5min.png')


# In[19]:


if ABIp == 'ACTP':
    bins = np.arange(0.5,6.5,1)
    ABI_pd.hist(bins=bins, figsize=(15,15))
    plt.savefig(save_pic_loc+ABIp+'_flash_5min.png')


# In[18]:


if ((GLMp == 'AFA') | (GLMp == 'MFA') | (GLMp == 'FE') | (GLMp == 'AGA')) | ((ABIp == 'CMIP13') & (GLMp == 'FED')):
    fig1 = plt.figure(figsize=(16, 8))
    fig1 = sns.violinplot(data=GLM_pd,cut=0)
    plt.title(v[GLMp][2]+' ('+GLMp+':5 min), '+case+' 12Z-12Z')
    plt.ylabel(v[GLMp][3])
    plt.grid(True)

    if GLMp == 'FE':
        plt.ylim(0,100)
    elif GLMp == 'FED':
        plt.ylim(0,10)

    plt.savefig(save_pic_loc+GLMp+'_5min.png')

