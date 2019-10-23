import sys
sys.path.insert(1, '/localdata/PyScripts/utilities/')
from fn_master import *
v = var_dict()


import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import seaborn as sns
import pandas as pd

args = sys.argv

file = args[1]

loc1 = '/localdata/cases/long_test/20190523/resampled_data/' 
loc2 = '/localdata/cases/long_test/20190524/resampled_data/' 
loc3 = '/localdata/cases/long_test/20190527/resampled_data/' 
loc4 = '/localdata/cases/long_test/20190528/resampled_data/' 
loc5 = '/localdata/cases/long_test/20190529/resampled_data/' 
save_loc = '/localdata/cases/long_test/combo/'

#Reading in the values and putting them all together
var1 = pd.read_pickle(loc1+file)
var2 = pd.read_pickle(loc2+file)
var3 = pd.read_pickle(loc3+file)
var4 = pd.read_pickle(loc4+file)
var5 = pd.read_pickle(loc5+file)
combo = np.concatenate((var1.values,var2.values,var3.values,var4.values,var5.values),axis=0)

#Grabbing the sample sizes of each bin
nan_truther = np.isnan(combo)
num_nums = np.ones((6))
for i in range(0,6,1):
    num_nums[i] = int(np.sum(~nan_truther[:,i]))
print (num_nums)

#Putting it all together into a new pandas dataframe
combo_pd = pd.DataFrame(combo,columns=['20-30 '+str(int(num_nums[0])),'30-40 '+str(int(num_nums[1])),'40-50 '+str(int(num_nums[2])),'50-60 '+str(int(num_nums[3])),'60-70 '+str(int(num_nums[4])),'70-80 '+str(int(num_nums[5]))])


#Color Formatting
sns.set(color_codes=True)
sns.set(style='ticks',font_scale=1.5)
colors = ['dodger blue', 'sea green', 'green', 'lime', 'yellow', 'orange']
xkcd = sns.xkcd_palette(colors)
sns.set_palette(xkcd)
#sns.palplot(sns.xkcd_palette(colors))



#Making the figure
fig1 = plt.figure(figsize=(16, 8))
fig1 = sns.violinplot(data=combo_pd,cut=0)
fig1.invert_yaxis()
plt.title('ABI Cloud Top Brightness Temperatures - Band 13 (Null Points)')
plt.ylabel('Brightness Temperatures')
plt.grid(True)

#plt.ylim(0,1)
plt.savefig(save_loc+'CMIP13_null_total.png')
#plt.show()


print ('For each bin:')
print (np.nanpercentile(combo,5,axis=0))
print ('Total:')
print (np.nanpercentile(combo,5))

  
