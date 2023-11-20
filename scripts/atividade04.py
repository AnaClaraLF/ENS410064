# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:18:43 2023

@author: 55489
"""
## dados de -entrada-
# ponto a ser investigado - Lages~ -27.81, -50.31
lai = -27.81
loi = -50.31
# quantos dias tem o ultimo mes da serie temporal
dd = 31
## dados de -entrada-

# pacotes
import os
import dask
import numpy as np
import numpy.ma as ma
import math
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import pandas as pd

# # abrir arquivos, juntar e salvar novo arq
# dask.config.set({"array.slicing.split_large_chunks": False})
# ds = xr.open_mfdataset('D:/ENS410064/Dados/brutos/earth/GLDAS2/*.nc4', combine='nested', concat_dim="time")
# ds.to_netcdf('D:/ENS410064/Dados/brutos/earth/GLDAS_NOAH025_MASUB.nc4')

# GLDAS2 documentation
# Evap_tavg Evapotranspiration kg m-2 s-1
# Albedo_inst Albedo %
# TVeg_tavg Transpiration W m-2
# Tair_f_inst Air temperature K
# https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/doc/README_GLDAS2.pdf
# Monthly average files contain straight averages of 3-hourly data, so each monthly
# average has units PER 3 HOURS. For example, total evapotranspiration
# (Evap_tavg) for April 1979 is the average 3-hour mean rate of evapotranspiration
# over all 3-hour intervals in April 1979. It is NOT the accumulated evapotranspiration
# in April 1979. To compute the latter, use this formula:
# Evap_tavg (April){kg/m2} =
# Evap_tavg (April){kg/m2/sec} * 10800{sec/3hr} * 8{3hr/day} * 30{days}

# conferir
data = nc.Dataset('D:/ENS410064/Dados/brutos/earth/GLDAS_NOAH025_MASUB.nc4')
data1 = nc.Dataset(r"D:\ENS410064\Dados\brutos\earth\GLDAS2\GLDAS_NOAH025_M.A202204.021.nc4.SUB.nc4")

# extraindo dados 
lon = data['lon'][:]
lat = data['lat'][:]
evap = data['Evap_tavg'][:]
alb = data['Albedo_inst'][:]
tmp = data['Tair_f_inst'][:]
tveg = data['Tveg_tavg'][:]
tempo = data['time']
diasm = np.diff(tempo) # num de dias em cada mes
diasm = np.append(diasm,dd) # falta o ultimo mes
# acumulado da evap mensal, kg de h2o/m² = mm de h2o
evapCum = ma.empty([284,20,27])
for i in range(len(diasm)):
    evapCum[i,:,:] = evap[i,:,:].__mul__(10800*8*diasm[i]) # em mm/mes

# no espaco, TEMPO ZERO, so pra ver
fig,ax = plt.subplots()
ax.pcolor(lon,lat,evap[0, :, :])

fig2,ax2 = plt.subplots()
ax2.pcolor(lon,lat,alb[0, :, :])

fig3,ax3 = plt.subplots()
ax3.pcolor(lon,lat,tmp[0, :, :])

fig4,ax4 = plt.subplots()
ax4.pcolor(lon,lat,tveg[0, :, :])

## Variabilidade temporal (EM UM PIXEL)
# media em ~lages -27.81, -50.31
# encontrar lat lon mais prox 
def find_nearest(array, values):
    idx = (np.abs(array-values)).argmin()
    return idx
yi=find_nearest(lat, lai)
xi=find_nearest(lon, loi)
#lon[0,17] -50,375 lat[7,0] -27,875

# dfs for descriptive stats on timeseries of lages
df_evap = pd.DataFrame(evapCum[:,yi,xi])
df_alb = pd.DataFrame(alb[:,yi,xi])
df_tmp = pd.DataFrame(tmp[:,yi,xi])
df_tveg = pd.DataFrame(tveg[:,yi,xi])

fig5, ax5 = plt.subplots()
ax5.hist(df_evap)
plt.figtext(0.1,0.5, df_evap.describe().to_string())
plt.figtext(0.75,0.5, df_evap.describe().loc[['mean','std']].to_string())

fig6, ax6 = plt.subplots()
ax6.hist(df_alb)
plt.figtext(0.1,0.5, df_alb.describe().to_string())
plt.figtext(0.75,0.5, df_alb.describe().loc[['mean','std']].to_string())

fig7, ax7 = plt.subplots()
ax7.hist(df_tveg)
plt.figtext(0.1,0.5, df_tveg.describe().to_string())
plt.figtext(0.75,0.5, df_tveg.describe().loc[['mean','std']].to_string())

fig8, ax8 = plt.subplots()
ax8.hist(df_tmp)
plt.figtext(0.1,0.5, df_tmp.describe().to_string())
plt.figtext(0.75,0.5, df_tmp.describe().loc[['mean','std']].to_string())

# variacao no tempo em lages
fig9,ax9 = plt.subplots()
ax9.plot(evapCum[:,yi,xi])


## Variabilidade espacial (TODOS PIXELS)
# MEDIAS
fig,ax = plt.subplots()
ax.pcolor(lon,lat,np.mean(evap[:, :, :], axis = 0))

fig2,ax2 = plt.subplots()
ax2.pcolor(lon,lat,np.mean(alb[:, :, :], axis = 0))

fig3,ax3 = plt.subplots()
ax3.pcolor(lon,lat,np.mean(tmp[:, :, :], axis = 0))

fig4,ax4 = plt.subplots()
ax4.pcolor(lon,lat,np.mean(tveg[:, :, :],axis = 0 ))


