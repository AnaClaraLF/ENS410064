# -*- coding: utf-8 -*-

# pacotes
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# caminho para meu arquivo com conc de o3
path = r'D:\ENS410064\Dados\brutos\BRAIN_ClippedCONC_O3_2019_07_02_11_to_2019_12_30_23.nc'

# abrir arquivo
data = nc.Dataset(path)
# extraindo dados de o3
o3 = data['O3'][:]
lat = data['LAT'][:]
lon = data['LON'][:]
tflag = data['TFLAG'][:]

#no tempo
fig,ax = plt.subplots()
ax.plot(o3[:,0,27,27])

#no espaco
fig2,ax2 = plt.subplots()
ax2.pcolor(lon,lat,np.mean(o3[0,0,:,:],axis = 0))
shp = 


