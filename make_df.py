#!/usr/bin/env python
# coding: utf-8




#This code is designed to:
#1) Make the GLM 5-minute composites
#2) Expand all GLM and ABI data into 20km resolution
#3) Put all of this data into a dataframe


#Sections 1 and 2 are the functions used
#Section 3 prepares to run those functions
#First it calls on the GLM_data and ABI_data function
#Those two functions call on the local functions: fivechecker, magic, GLMdf, nan5, ACMapply, ACHAapply, ABIdf
#They also call on the remote functions from fn_master.py: resample and gridexpand

#A key upgrade over the previous workflow is the ability to work around gaps in the datasets if they exist



import pandas as pd
import netCDF4 as nc
import numpy as np
import os
import sys
import glob
from datetime import datetime
sys.path.insert(1, '/localdata/PyScripts/utilities/')
from fn_master import *



#=======================================================================================================
# Function Land
#=======================================================================================================


# # Section 1
# The sub functions that are used by GLM_data and ABI_data




#This function takes in the file locations, names, and current index
#Then 'globs' together the next five minutes worth of files

def fivechecker(floc,flist,i):
    files = ()
    
    a = glob.glob(floc+flist[i])
    files = np.append(files,a)
    b = glob.glob(floc+flist[i+1])
    files = np.append(files,b)
    c = glob.glob(floc+flist[i+2])
    files = np.append(files,c)
    d = glob.glob(floc+flist[i+3])
    files = np.append(files,d)
    e = glob.glob(floc+flist[i+4])
    files = np.append(files,e)
    
    return files





#This function creates the five minute composites for the GLM data

def magic(files,variable,command):
    #Opening up each of the files
    file5 = nc.Dataset(files[0],'r')
    file4 = nc.Dataset(files[1],'r')
    file3 = nc.Dataset(files[2],'r')
    file2 = nc.Dataset(files[3],'r')
    file1 = nc.Dataset(files[4],'r')
    
    #Extracting the variable
    var5 = file5.variables[variable][:,:].astype(np.float32)
    var4 = file4.variables[variable][:,:].astype(np.float32)
    var3 = file3.variables[variable][:,:].astype(np.float32)
    var2 = file2.variables[variable][:,:].astype(np.float32)
    var1 = file1.variables[variable][:,:].astype(np.float32)
    
    file5.close()
    file4.close()
    file3.close()
    file2.close()
    file1.close()

    #Filling the masked arrays with nans to do the calculations
    var5 = np.ma.filled(var5,fill_value=np.nan)
    var4 = np.ma.filled(var4,fill_value=np.nan)
    var3 = np.ma.filled(var3,fill_value=np.nan)
    var2 = np.ma.filled(var2,fill_value=np.nan)
    var1 = np.ma.filled(var1,fill_value=np.nan)
    
    
    #Stacking all the variables
    stack_var = np.stack((var5,var4,var3,var2,var1),axis=2)
    
    #Making the composite or sums
    if command == 'max':
        final_var = np.nanmax(stack_var,axis=2)
    elif command == 'sum':
        final_var = np.nansum(stack_var,axis=2)
    elif command == 'min':
        final_var = np.nanmin(stack_var,axis=2)
    elif command == 'avg':
        final_var = np.nanmean(stack_var,axis=2)
    
    masker1 = np.isnan(final_var)
    final_var[masker1] = 0
    
    masker2 = final_var == 0
    final_var = np.ma.masked_array(final_var,mask=masker2,fill_value=-999)
    
    return final_var





#This function creates the GLM dataframe

def GLMdf(FED,AFA,TOE,MFA,ilist,i):
    #Flattening the arrays to match the x/y coordinate combinations
    FEDf = FED.flatten()
    AFAf = AFA.flatten()
    TOEf = TOE.flatten()
    MFAf = MFA.flatten()
    print ('GLM '+ilist[i+5]) # Printing the time
    time = np.broadcast_to(ilist[i+5],len(xxf))
    
    #Creating a dummy series to assign the multiindex to
    df = pd.Series(dummy,name='Dummy')
    mi = pd.MultiIndex.from_arrays([time[:],xxf,yyf])
    df = df.reindex(mi)
    
    #Putting all of the data into a dataframe
    newdf = pd.DataFrame({'Dummy':df,
                     'FED':FEDf,
                     'AFA':AFAf,
                     'MFA':MFAf,
                     'TOE':TOEf,
                     'angle':angle})
    
    return newdf.drop(columns='Dummy') #Removing the dummy data before sending it back





#This function creates a dataframe full of nans for when the GLM 5-minute composites can't be made

def nan5(ilist,i):
    print ('*********MISSING**********')
    print (ilist[i+5])
    print ('**************************')
    
    time = np.broadcast_to(ilist[i+5],len(xxf))
    nans = np.ones(len(xxf))*np.nan
    
    df = pd.Series(nans)
    mi = pd.MultiIndex.from_arrays([time,xxf,yyf])
    df = df.reindex(mi)
    
    newdf = pd.DataFrame({'FED':df,
                     'AFA':nans,
                     'MFA':nans,
                     'TOE':nans,
                     'angle':angle})
    return newdf





#This function processes the CMIP data and applies the ACM to avoid the ground points

def ACMapply(ACM_files,CMIP_files):
    #Loading/Extracting the ACM and CMIPdata
    ACMd = nc.Dataset(ACM_files[0],'r')
    CMIPd = nc.Dataset(CMIP_files[0],'r')
    ACM_var = ACMd.variables['BCM'][:,:].astype(np.float32)
    CMIP_var = CMIPd.variables['CMI'][:,:].astype(np.float32)
    CMIP_var = np.ma.filled(CMIP_var,fill_value=np.nan)
    ACMd.close()
    CMIPd.close()
    

    #Applying the mask
    CMIP = CMIP_var*ACM_var

    #Turning the zero values from the mask into nans
    zeros = CMIP == 0
    CMIP[zeros] = np.nan
    
    CMIP = gridexpand(CMIP,'min')
    
    return (CMIP)





#This function processes the ACHA data

def ACHAapply(ACHA_files):
    #Loading the data
    ACHAd = nc.Dataset(ACHA_files[0],'r')
    ACHA_var = ACHAd.variables['HT'][:,:]
    ACHA_var = np.ma.filled(ACHA_var,fill_value=np.nan)
    
    ACHAd.close()
    ACHA_var = resample(ACHA_var) #resampled to 2km 
    ACHA = gridexpand(ACHA_var,'max') #expanded to 20km
    
    smol = ACHA < 1000
    ACHA[smol] = np.nan
    
    return (ACHA)





#This function creates the creates the ABI dataframes

def ABIdf(CMIP,ACHA,ilist,i):
    CMIPf = CMIP.flatten()
    ACHAf = ACHA.flatten()
    
    print ('ABI '+ilist[i+5])
    time = np.broadcast_to(ilist[i+5],len(xxf))
    
    df = pd.Series(dummy,name='Dummy')
    mi = pd.MultiIndex.from_arrays([time[:],xxf,yyf])
    df = df.reindex(mi)
    
    newdf = pd.DataFrame({'Dummy':df,
                          'CMIP':CMIPf
                         ,'ACHA':ACHAf})
           
    return newdf.drop(columns='Dummy')


# # Section 2
# The 'driver' functions called below




#Funciton that controls the GLM data processing

def GLM_data(tlist,GLM_loc,GLM_list,ilist):
    index = np.arange(0,len(tlist))
    
    A = ['Null']
    df = pd.Series([np.nan])
    mi = pd.MultiIndex.from_arrays([A,A,A])
    GLM_df = df.reindex(mi)
    
    #Loop that checks if the files exist to make the composite, and then creates the composite dataframe
    for i in index[:-1:5]:
        #checks if the files exist
        files = fivechecker(GLM_loc,GLM_list,i)
        
        if len(files) == 5:
            #Gathering the 5-minute composites
            FED = magic(files,'flash_extent_density','sum')
            AFA = magic(files,'average_flash_area','avg')
            TOE = magic(files,'total_energy','sum')*10**6
            MFA = magic(files,'minimum_flash_area','min')
            #Expanding the resolution to 20km
            FED = gridexpand(FED,'max')
            AFA = gridexpand(AFA,'avg')
            TOE = gridexpand(TOE ,'max')
            MFA = gridexpand(MFA,'min')
            
            df = GLMdf(FED,AFA,TOE,MFA,ilist,i) #Funciton that puts the data into the dataframe
            GLM_df = pd.concat((GLM_df,df),axis=0,sort=True)
            
        else:
            df = nan5(ilist,i)
            GLM_df = pd.concat((GLM_df,df),axis=0,sort=True)
    
    print('GLM DataFrame Created')
    return GLM_df
        





#Function that controls the ABI data processing

def ABI_data(ACHA_loc,ACHA_list,ACM_loc,ACM_list,CMIP_loc,CMIP_list,ilist,tlist):
    index = np.arange(0,len(tlist))
    
    A = ['Null']
    df = pd.Series([np.nan])
    mi = pd.MultiIndex.from_arrays([A,A,A])
    ABI_df = df.reindex(mi)

    for i in index[:-1:5]:
        #Checking the files within the five minutes that coincide with the GLM data
        ACHA_files = fivechecker(ACHA_loc,ACHA_list,i)
        ACM_files = fivechecker(ACM_loc,ACM_list,i)
        CMIP_files = fivechecker(CMIP_loc,CMIP_list,i)
        if (len(CMIP_files)==1) & (len(ACM_files)==1):
            CMIP_data = ACMapply(ACM_files,CMIP_files)
        else:
            CMIP_data = np.ones(xxf.shape)*np.nan
            print ('********NAN CMIP*********')
            
        if (len(ACHA_files)==1):
            ACHA_data = ACHAapply(ACHA_files)
        else:
            ACHA_data = np.ones(xxf.shape)*np.nan
            print ('********NAN ACHA*********')
            
        df = ABIdf(CMIP_data,ACHA_data,ilist,i)
        
        ABI_df = pd.concat((ABI_df,df),axis=0,sort=True)

    gprint ('ABI Dataframe Created')
    return (ABI_df)



#=======================================================================================================
# Run like this: python make_df.py long_test/20190527 20190527 20190528 long_test
#=======================================================================================================



# # Section 3
# Setting up the data to run the driver functions



#Reading in the arguments to use
args = sys.argv
#args = ['long_test/20190525','20190525','20190526','long_test']#casename,start date, end date, name

case = args[1]
sdate = args[2]
edate = args[3]
name = args[4]


#Getting the locations of the data sources
GLM_loc = '/localdata/cases/'+case+'/GLM_grids/'
ACHA_loc = '/localdata/cases/'+case+'/ABI/ACHAC/'
CMIP_loc = '/localdata/cases/'+case+'/ABI/CMIPC13/'
ACM_loc = '/localdata/cases/'+case+'/ABI/ACMC/'
save_loc = '/localdata/cases/'+name+'/data/'


#Loading the x and y grids
y = np.load('/localdata/coordinates/20km_y.npy')
x = np.load('/localdata/coordinates/20km_x.npy')
xx, yy = np.meshgrid(x,y)
del (x,y)
xxf = xx.flatten()
yyf = yy.flatten()
del (xx,yy)
dummy = np.ones(len(xxf))*np.nan #A dummy array we use to prefill the dataframes


#Loading the viewing angle data at a 20 km resolution, and assigning NaNs to angles calculated for in space (>76.184718)
angle = np.load('/localdata/coordinates/GLM/viewing_angle2.npy')[::10,::10].flatten()
off = angle >= 76.184718
angle[off] = np.nan


#Creating the list of times which we will use to create our filenames to check later
stime = datetime.strptime(sdate,'%Y%m%d')
etime = datetime.strptime(edate,'%Y%m%d')
tlist = pd.date_range(start=stime,end=etime,freq='T').to_pydatetime().tolist()

index = np.arange(0,len(tlist))


GLM_list = np.empty(0) #List that will be filled with the GLM filenames
ACHA_list = np.empty(0) #List that will be filled with the ACHA filenames
CMIP_list = np.empty(0) #List that will be filled with the ACHA filenames
ACM_list = np.empty(0) #List that will be filled with the ACHA filenames
ilist = np.empty(0) #List that will be filled with the times for the index


#Loop that creates the file lists and time index
for i in index:
    y = datetime.strftime(tlist[i],'%Y')
    j = datetime.strftime(tlist[i],'%j')
    h = datetime.strftime(tlist[i],'%H')
    M = datetime.strftime(tlist[i],'%M')
    d = datetime.strftime(tlist[i],'%d')
    m = datetime.strftime(tlist[i],'%m')

    GLM_str = 'OR_GLM-L2-GLMC-M3_G16_s'+y+j+h+M+'*.nc'
    GLM_list = np.append(GLM_list,GLM_str)
    
    ACHA_str = 'OR_ABI-L2-ACHAC-M6_G16_s'+y+j+h+M+'*.nc'
    ACHA_list = np.append(ACHA_list,ACHA_str)
    
    ACM_str = 'OR_ABI-L2-ACMC-M6_G16_s'+y+j+h+M+'*.nc'
    ACM_list = np.append(ACM_list,ACM_str)
    
    CMIP_str = 'OR_ABI-L2-CMIPC-M6C13_G16_s'+y+j+h+M+'*.nc'
    CMIP_list = np.append(CMIP_list,CMIP_str)

    istr = y+m+d+'-'+h+M
    ilist = np.append(ilist,istr)
    

# Calling the main driver functions GLM_data and ABI_data
GLM_df = GLM_data(tlist,GLM_loc,GLM_list[:-1],ilist)
ABI_df = ABI_data(ACHA_loc,ACHA_list,ACM_loc,ACM_list,CMIP_loc,CMIP_list,ilist,tlist)

del (GLM_list,ACHA_list,CMIP_list,ACM_list,i_list,y,j,h,M,d,m)

#Dropping the Null indicies and also the placeholder column we used to make the empty dataframe
ABI_df = ABI_df.drop(index=['Null'],columns=0)
GLM_df = GLM_df.drop(index=['Null'],columns=0)



#Putting the two dataframes together
dataframe = pd.concat((ABI_df,GLM_df),axis=0,sort=True).astype('float32')
del (ABI_df,GLM_df)

#Saving the dataframe
if not os.path.exists(save_loc):
    os.makedirs(save_loc)  
dataframe.to_pickle(save_loc+sdate+'.pkl')
