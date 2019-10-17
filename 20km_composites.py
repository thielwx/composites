#!/usr/bin/env python
# coding: utf-8



#For importig the fn_master module
import sys
sys.path.insert(1, '/localdata/PyScripts/utilities/')
from fn_master import *
v = var_dict()
#Importing all the other dictionaries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import seaborn as sns



#Color Formatting
sns.set(color_codes=True)
sns.set(style='ticks',font_scale=1.5)
colors = ['dodger blue', 'sea green', 'green', 'lime', 'yellow', 'orange']
xkcd = sns.xkcd_palette(colors)
sns.set_palette(xkcd)
#sns.palplot(sns.xkcd_palette(colors))


#==========================================================
# Function land
#==========================================================

#Loading the ABI files
def ABI_fileload(file,var):
    
    dataset = nc.Dataset(file, 'r')
    ABI_x = dataset.variables['x'][:] 
    ABI_y = dataset.variables['y'][:]
    ABI_var = dataset.variables[var][:,:]
    ABI_var = np.ma.filled(ABI_var,fill_value=np.nan)
    
    ABI_time = datetime.strptime(dataset.time_coverage_end, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    dataset.close()
    
    return ABI_x,ABI_y,ABI_var,ABI_time



#Loading the GLM files
def GLM_fileload(file,var):
    
    dataset = nc.Dataset(file,'r')
    GLM_var = dataset.variables[var][:,:]
    GLM_var = np.ma.filled(GLM_var,fill_value=np.nan)
    
    return GLM_var



#Applying the ABI Clear Sky Mask to the CMIP13 data
def ACM_apply(i,var):
    ACMd = nc.Dataset('/localdata/cases/'+case+'/ABI/ACMC/'+ACM_filelist[i],'r')
    ACMv = ACMd.variables['BCM'][:,:]
    mvar = ACMv*var
    
    zeros = mvar == 0
    mvar[zeros] = np.nan
    ACMd.close()
    return mvar



#Separating the data by the six defined viewing angle bins
def separator(var):
    
    var_a = var[a]
    var_b = var[b]
    var_c = var[c]
    var_d = var[d]
    var_e = var[e]
    var_f = var[f]
    
    var_a_num = var_a[~np.isnan(var_a)]
    var_b_num = var_b[~np.isnan(var_b)]
    var_c_num = var_c[~np.isnan(var_c)]
    var_d_num = var_d[~np.isnan(var_d)]
    var_e_num = var_e[~np.isnan(var_e)]
    var_f_num = var_f[~np.isnan(var_f)]
    
    aa = len(var_a_num)
    bb = len(var_b_num)
    cc = len(var_c_num)
    dd = len(var_d_num)
    ee = len(var_e_num)
    ff = len(var_f_num)
    print (aa,bb,cc,dd,ee,ff)
    i = np.max([aa,bb,cc,dd,ee,ff])
    print (i)
    Data_20_var = np.concatenate((np.ones(i-aa)*np.nan,var_a_num))
    Data_30_var = np.concatenate((np.ones(i-bb)*np.nan,var_b_num))
    Data_40_var = np.concatenate((np.ones(i-cc)*np.nan,var_c_num))
    Data_50_var = np.concatenate((np.ones(i-dd)*np.nan,var_d_num))
    Data_60_var = np.concatenate((np.ones(i-ee)*np.nan,var_e_num))
    Data_70_var = np.concatenate((np.ones(i-ff)*np.nan,var_f_num))
    Data_var = np.stack((Data_20_var,Data_30_var,Data_40_var,Data_50_var,Data_60_var,Data_70_var),axis=1)
    var_pd = pd.DataFrame(Data_var,columns=['20-30 '+str(aa),'30-40 '+str(bb),'40-50 '+str(cc),'50-60 '+str(dd),'60-70 '+str(ee),'70-80 '+str(ff)])

    return var_pd

#============================================================
# Run like this: python 20km_composites.py CMIP13 FED long_test/20190527
#===========================================================



args = sys.argv

ABIp = args[1]
GLMp = args[2]
case = args[3]




#Initializing the data
ABI_loc = '/localdata/cases/'+case+'/ABI/'+v[ABIp][0]+'/'
GLM_loc = '/localdata/cases/'+case+'/GLM5/'
save_file_loc = '/localdata/cases/'+case+'/resampled_data/'
save_pic_loc = '/localdata/cases/'+case+'/resampled_pics/'

if not os.path.exists(save_pic_loc):
            os.makedirs(save_pic_loc)
if not os.path.exists(save_file_loc):
            os.makedirs(save_file_loc)

ABI_filelist = sorted(os.listdir(ABI_loc))
ACM_filelist = sorted(os.listdir('/localdata/cases/'+case+'/ABI/ACMC/'))
GLM_filelist = sorted(os.listdir(GLM_loc))

filelist_index = np.arange(0,len(ABI_filelist))




GLM_composite = np.zeros((1,1500,2500)).astype(np.float16)
ABI_composite = np.zeros((1,1500,2500)).astype(np.float16)




GLM_composite = np.zeros((1,150,250)).astype(np.float16)
ABI_composite = np.zeros((1,150,250)).astype(np.float16)

for i in filelist_index:
    #Loading in the ABI data
    ABI_x,ABI_y,ABI_var,ABI_time = ABI_fileload(ABI_loc+ABI_filelist[i],v[ABIp][1])
    #Loading in the GLM data
    GLM_var = GLM_fileload(GLM_loc+GLM_filelist[i],v[GLMp][1])
    print (ABI_time)
    
    #Resampling the ABI data to match the GLM grids if needed or applying the clear sky mask
    if (ABIp == 'CMIP13') | (ABIp == 'ACTP'):
        ABI_new = ACM_apply(i,ABI_var)
    else:
        ABI_new = resample(ABI_var)
    
    #Back the resolution out to 20 km points (150,250)    
    ABI_expand = gridexpand(ABI_new,v[ABIp][4])
    GLM_expand = gridexpand(GLM_var,v[GLMp][4])
    
    ABI_processed = np.expand_dims(ABI_expand,axis=0)
    GLM_processed = np.expand_dims(GLM_expand,axis=0)
    
    ABI_composite = np.append(ABI_composite,ABI_processed,axis=0)
    GLM_composite = np.append(GLM_composite,GLM_processed,axis=0)

    
ABI_dataset = ABI_composite[1::,:,:]
GLM_dataset = GLM_composite[1::,:,:]

del(ABI_composite,GLM_composite,ABI_processed,GLM_processed,ABI_expand,GLM_expand)




#Finding all the nans
ABInans = np.isnan(ABI_dataset)
GLMnans = np.isnan(GLM_dataset)

#Null and flash datasets
ABI_compositef = ABI_dataset.copy()
ABI_compositen = ABI_dataset.copy()
GLM_compositef = GLM_dataset.copy()

#Applying nans at grid points from one dataset to another and vice versa
#Also taking where nans don't exist (the data points) from the GLM and applying that to the ABI null dataset
ABI_compositef[GLMnans] = np.nan
ABI_compositen[~GLMnans] = np.nan
GLM_compositef[ABInans] = np.nan




angle = np.load('/localdata/coordinates/GLM/viewing_angle2.npy')[::10,::10]
#Setting up boolean arrays to extract angles
f = (angle>=70.0) & (angle<80.0)
e = (angle>=60.0) & (angle<70.0)
d = (angle>=50.0) & (angle<60.0)
c = (angle>=40.0) & (angle<50.0)
b = (angle>=30.0) & (angle<40.0)
a = (angle>=20.0) & (angle<30.0)

a = np.broadcast_to(a,GLM_compositef.shape)
b = np.broadcast_to(b,GLM_compositef.shape)
c = np.broadcast_to(c,GLM_compositef.shape)
d = np.broadcast_to(d,GLM_compositef.shape)
e = np.broadcast_to(e,GLM_compositef.shape)
f = np.broadcast_to(f,GLM_compositef.shape)


#Running the separator function for the three functions
ABIf_pd = separator(ABI_compositef)
ABIn_pd = separator(ABI_compositen)
GLM_pd = separator(GLM_compositef)


if GLMp == 'FE':
    GLM_pd *= 10**6
    




#Saving the data as a pickle files, that way we can reload and display them faster in the future

if (ABIp == 'CMIP13') & (GLMp == 'FED'):
    ABIf_pd.to_pickle(save_file_loc+ABIp+'_flash_20km.pkl')
    ABIn_pd.to_pickle(save_file_loc+ABIp+'_null_20km.pkl')
    GLM_pd.to_pickle(save_file_loc+GLMp+'_20km.pkl')
elif (ABIp == 'CMIP13'):
    GLM_pd.to_pickle(save_file_loc+GLMp+'_20km.pkl')
elif (GLMp =='FED'):
    ABIf_pd.to_pickle(save_file_loc+ABIp+'_flash_20km.pkl')
    ABIn_pd.to_pickle(save_file_loc+ABIp+'_null_20km.pkl')




if ((ABIp == 'CTP') | (ABIp == 'ACHA')) | ((ABIp == 'CMIP13') & (GLMp == 'FED')):
    fig1 = plt.figure(figsize=(16, 8))
    fig1 = sns.violinplot(data=ABIf_pd,cut=0)
    #fig1.tick_params(labelsize=5)
    if (ABIp == 'CMIP13') | (ABIp == 'CTP'):
        fig1.invert_yaxis() #Only for CTP and CMIP13

    plt.title(v[ABIp][2]+' ('+ABIp+': Flash Points), '+case+' 0Z-0Z')
    plt.ylabel(v[ABIp][3])
    plt.grid(True)
    
    plt.savefig(save_pic_loc+ABIp+'_flash_20km.png')




if ((ABIp == 'CTP') | (ABIp == 'ACHA')) | ((ABIp == 'CMIP13') & (GLMp == 'FED')):
    fig1 = plt.figure(figsize=(16, 8))
    fig1 = sns.violinplot(data=ABIn_pd,cut=0)
    #fig1.tick_params(labelsize=5)
    if (ABIp == 'CMIP13') | (ABIp == 'CTP'):
        fig1.invert_yaxis() #Only for CTP and CMIP13

    plt.title(v[ABIp][2]+' ('+ABIp+' : Null Points), '+case+' 0Z-0Z')
    plt.ylabel(v[ABIp][3])
    plt.grid(True)
    
    plt.savefig(save_pic_loc+ABIp+'_null_20km.png')




if ((GLMp == 'AFA') | (GLMp == 'MFA') | (GLMp == 'FE')) | ((ABIp == 'CMIP13') & (GLMp == 'FED')):
    fig1 = plt.figure(figsize=(16, 8))
    fig1 = sns.violinplot(data=GLM_pd,cut=0)
    plt.title(v[GLMp][2]+' ('+GLMp+'), '+case+' 0Z-0Z')
    plt.ylabel(v[GLMp][3])
    plt.grid(True)

    if GLMp == 'FE':
        plt.ylim(0,100)
    elif GLMp == 'FED':
        plt.ylim(0,10)
        plt.grid(True)

    plt.savefig(save_pic_loc+GLMp+'_20km.png')






