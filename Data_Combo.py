
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import seaborn as sns
import pandas as pd
import sys

args = sys.argv

file = args[1]

loc1 = '/localdata/cases/20190414/resampled_data/' 
loc2 = '/localdata/cases/20190527/resampled_data/'
loc3 = '/localdata/cases/20190620/resampled_data/'
save_loc = '/localdata/cases/combo_case/'

var1 = pd.read_pickle(loc1+file)
var2 = pd.read_pickle(loc2+file)
var3 = pd.read_pickle(loc3+file)
combo = np.concatenate((var1.values,var2.values,var3.values),axis=0)

print ('For each bin:')
print (np.nanpercentile(combo,5,axis=0))
print ('Total:')
print (np.nanpercentile(combo,5))

  
