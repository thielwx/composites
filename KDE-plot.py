import netCDF4 as nc
import numpy as np
from scipy import interpolate
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj
from pyresample import image,geometry,SwathDefinition
import sys

#=========================
#RUN LIKE THIS
#python KDE-plot.py varp GLMp case
#=========================


args = sys.argv

varp = args[1]
GLMp = args[2]
case = args[3]



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

if (GLMp == 'AFA') | (GLMp == 'FED') | (GLMp == 'FE') | (GLMp == 'AGA'):
    GLM_kind = 'max'
else:
    GLM_kind = 'min'

if (varp == 'ACHA'):
    var_kind = 'max'
else:
    var_kind = 'min'



def gridfun(data,kind):
    latlen = len(lat2[::10])
    lonlen = len(lon2[::10])
    latpts = range(0,latlen,1)
    lonpts = range(0,lonlen,1)

    nloc = data == 0
    data[nloc] = np.nan

    new_grid = np.ones((latlen,lonlen))
    for lat in latpts:
        for lon in lonpts:
            section = data[lat*10:(lat+1)*10,lon*10:(lon+1)*10]
            if kind == 'max':
                new_grid[lat,lon] = np.nanmax(section)
            elif kind == 'min':
                new_grid[lat,lon] = np.nanmin(section)
    ntruther = np.isnan(new_grid)
    new_grid[ntruther] = 0.0   
    return new_grid



#Data Locations
var_loc = '/localdata/cases/'+case+'/ABI/'+v[varp][0]+'/'
GLM_loc = '/localdata/cases/'+case+'/GLM5/'
save_pic_loc = '/localdata/cases/'+case+'/JDP_pics/'
save_file_loc = '/localdata/cases/'+case+'/JDP_data/'

GLM_filelist = sorted(os.listdir(GLM_loc))
var_filelist = sorted(os.listdir(var_loc))
halfway = len(GLM_filelist) / 2
index_mover = np.arange(0,len(GLM_filelist),1)
print (len(GLM_filelist),len(var_filelist))



#Setting up the resampled swaths
lat2 = np.load('/localdata/coordinates/2km_lat.npy')
lon2 = np.load('/localdata/coordinates/2km_lon.npy')
lat2_grid = np.load('/localdata/coordinates/2km_lat_grid.npy')
lon2_grid = np.load('/localdata/coordinates/2km_lon_grid.npy')
lat10= np.load('/localdata/coordinates/10km_lat.npy')
lon10= np.load('/localdata/coordinates/10km_lon.npy')
swath_def10 = SwathDefinition(lons=lon10,lats=lat10)
swath_def2 = SwathDefinition(lons=lon2_grid,lats=lat2_grid)


GLM_composite_a = np.zeros((1,150,250)).astype(np.float16)
GLM_composite_b = np.zeros((1,150,250)).astype(np.float16)
var_composite_a = np.zeros((1,150,250)).astype(np.float16)
var_composite_b = np.zeros((1,150,250)).astype(np.float16)



for i in index_mover:
    GLM_file = nc.Dataset(GLM_loc+GLM_filelist[i],'r')
    var_file = nc.Dataset(var_loc+var_filelist[i],'r')
    GLM_var = np.ma.filled(GLM_file.variables[v[GLMp][1]][:,:],fill_value=0.0)
    var_var = np.ma.filled(var_file.variables[v[varp][1]][:,:],fill_value=0.0)
    
    if varp == 'ACHA':
        swath_con = image.ImageContainerNearest(var_var,swath_def10,radius_of_influence=10000)
        swath_resampled = swath_con.resample(swath_def2)
        var_new = swath_resampled.image_data
    else:
        var_new = var_var #Use only for the CMIP data
    
    GLM_big = gridfun(GLM_var,GLM_kind)
    var_big = gridfun(var_new,var_kind)

    #Creating the composites
    var_final = np.expand_dims(var_big,axis=0)
    GLM_final = np.expand_dims(GLM_big,axis=0)

    print (GLM_filelist[i])
    
        
    if GLM_composite_a.shape[0] < halfway:
        var_composite_a = np.append(var_composite_a,var_final,axis=0)
        GLM_composite_a = np.append(GLM_composite_a,GLM_final,axis=0)
    else:
        var_composite_b = np.append(var_composite_b,var_final,axis=0)
        GLM_composite_b = np.append(GLM_composite_b,GLM_final,axis=0)

    var_file.close()
    GLM_file.close()


var_composite = np.append(var_composite_a,var_composite_b,axis=0)
GLM_composite = np.append(GLM_composite_a,GLM_composite_b,axis=0)
del(var_composite_a,var_composite_b,GLM_composite_a,GLM_composite_b)



#The values for the var_composite_loc can also be used for thresholding the var products
GLM_overlap = GLM_composite.copy()
var_overlap = var_composite.copy()
#Boolean array where GLM/var data are zero
GLM_composite_loc = GLM_composite == 0.0
var_composite_loc = var_composite == 0.0
#Applying from one dataset to the other
GLM_overlap[var_composite_loc] = 0.0
GLM_overlap[GLM_composite_loc] = 0.0 #we have to apply the GLM boolean for any values over zero
var_overlap[GLM_composite_loc] = 0.0
#overlap is the main dataset now
del(GLM_composite_loc,var_composite_loc)



angle = np.load('/localdata/coordinates/GLM/viewing_angle2.npy')[::10,::10]
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

var_a = var_overlap[a]
var_b = var_overlap[b]
var_c = var_overlap[c]
var_d = var_overlap[d]
var_e = var_overlap[e]
var_f = var_overlap[f]
print ('Boolean arrays applied')

GLM_a_num = GLM_a[np.nonzero(GLM_a)]
GLM_b_num = GLM_b[np.nonzero(GLM_b)]
GLM_c_num = GLM_c[np.nonzero(GLM_c)]
GLM_d_num = GLM_d[np.nonzero(GLM_d)]
GLM_e_num = GLM_e[np.nonzero(GLM_e)]
GLM_f_num = GLM_f[np.nonzero(GLM_f)]
print ('GLM Data Extracted')
var_a_num = var_a[np.nonzero(var_a)]
var_b_num = var_b[np.nonzero(var_b)]
var_c_num = var_c[np.nonzero(var_c)]
var_d_num = var_d[np.nonzero(var_d)]
var_e_num = var_e[np.nonzero(var_e)]
var_f_num = var_f[np.nonzero(var_f)]
print ('var Data Extracted')

del(a,b,c,d,e,f,var_a,var_b,var_c,var_d,var_e,var_f,GLM_a,GLM_b,GLM_c,GLM_d,GLM_e,GLM_f)



GLM_together = np.concatenate((GLM_b_num,GLM_c_num),axis=0)
var_together = np.concatenate((var_b_num,var_c_num),axis=0)
n = str(len(GLM_together))
print (n)



GLM_pd = pd.Series(GLM_together,name=v[GLMp][3])
var_pd = pd.Series(var_together,name=v[varp][3])


if GLMp == 'FE':
    GLM_pd *= 10**6


a = sns.jointplot(GLM_pd,var_pd, height=10, color='blue', alpha=0.05)
fig = a.fig
fig.suptitle(GLMp+'-'+varp+' Joint-Distribution Plot, 30-50 deg, '+case+', n='+n, fontsize=15)
a.savefig(save_pic_loc+GLMp+'-'+varp+'-reg20.png')



pd_together = pd.concat((GLM_pd,var_pd),axis=1)

pd_together.to_pickle(save_file_loc+'-'+GLMp+'-'+varp+'-20km.pkl')
