#!/usr/bin/env python
# coding: utf-8



#This script is the insane attempt to run entire days separately and then put them together 
#into weekly/monthly/seasonal composites

#The script will run on a day by day basis, and be made so that each command has an assocaited function.
#So if I already have the data I will be able to only run certain parts



import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import datetime
import pandas as pd
import subprocess as sp

import sys
sys.path.insert(1, '/localdata/PyScripts/utilities')
from fn_master import *
v = var_dict()


#==========================================================
# Function land
#==========================================================



# # 3) Create GLM 5-minute composites


def glm5(case):
    
    cmd = 'python /localdata/PyScripts/utilities/GLM5_min.py '+case
    p = sp.Popen(cmd,shell=True)
    p.wait()
    
    return 0


# # 4) Create the daily binned composites that merge ABI and GLM data


def binned(case):
    
    for i in ['FED','AFA','MFA','FE']: 
        cmd = 'python /localdata/PyScripts/composites/20km_composites.py CMIP13 '+i+' '+case
        p = sp.Popen(cmd,shell=True)
        p.wait()
        
    for i in ['ACHA']:
        cmd = 'python /localdata/PyScripts/composites/20km_composites.py '+i+' FED '+case
        p = sp.Popen(cmd,shell=True)
        p.wait()
        
    return 0




#============================================================
# Run like this: python Long_Control3.py 'casename' YYYY/MM/DD(start) YYYY/MM/DD(end)
#===========================================================





args = sys.argv
#args = ['long_test','2019/05/27','2019/05/29']

case = args[1]
start_date = args[2]
end_date = args[3]




#Getting list of dates to pull from
datetime_list = pd.date_range(start=start_date,end=end_date,freq='D').to_pydatetime().tolist()

#Converting this into a list of dates that we can use later in the functions
start_list = []
end_list = []
timedelta = datetime.timedelta(days=1)

#For loop to create the start and end lists
for i in datetime_list:
    
    timestr = i.strftime('%Y%m%d')
    start_list.append(timestr)
    
    endtime = i + timedelta
    endstr = endtime.strftime('%Y%m%d')
    end_list.append(endstr)
    
indexes = np.arange(0,len(start_list))





for i in indexes:
    #glm5(case+'/'+start_list[i])
    binned(case+'/'+start_list[i])
    
