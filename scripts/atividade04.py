# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:18:43 2023

@author: 55489
"""
## dados de -entrada-
# ponto a ser investigado - Lages~ -27.81, -50.31 - Coxilha -28,20 -50,33
lai = -28.20
loi = -50.33
# quantos dias tem o ultimo mes da serie temporal
dd = 31
## dados de -entrada-

# pacotes
#import os
#import dask
import numpy as np
import numpy.ma as ma
#import math
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import pymannkendall as mk

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
#data = nc.Dataset(r'C:\Users\anafranco\Documents\ENS410064\GLDAS_NOAH025_MASUB.nc4')
#data1 = nc.Dataset(r"D:\ENS410064\Dados\brutos\earth\GLDAS2\GLDAS_NOAH025_M.A202204.021.nc4.SUB.nc4")

#data2 = data.subset(time=[0,283]) # descobrir como fazer subsets

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

## Variabilidade espacial (TODOS PIXELS)
# MEDIAS 
fig,ax = plt.subplots()
im = ax.pcolor(lon,lat,np.mean(evapCum[:, :, :], axis = 0))
plt.title('Evapotranspiracao')
fig.colorbar(im,ax=ax)

fig2,ax2 = plt.subplots()
im2 = ax2.pcolor(lon,lat,np.mean(alb[:, :, :], axis = 0))
plt.title('Albedo')
fig.colorbar(im2,ax=ax2)

fig3,ax3 = plt.subplots()
im3 = ax3.pcolor(lon,lat,np.mean(tmp[:, :, :], axis = 0))
plt.title('Temperatura do ar')
fig.colorbar(im3,ax=ax3)

fig4,ax4 = plt.subplots()
im4 = ax4.pcolor(lon,lat,np.mean(tveg[:, :, :],axis = 0 ))
plt.title('Transpiracao Veg')
fig.colorbar(im4,ax=ax4)

## Variabilidade espacial (TODOS PIXELS)
# por mes
mes = ['janeiro','fevereiro','marco','abril', 'maio','junho','julho','agosto',
       'setembro', 'outubro','novembro','dezembro']
mynorm = plt.Normalize(vmin=0, vmax=180) # manter escala
fig5,ax5 = plt.subplots(3,4,sharex=True)
ax5 = ax5.flatten() # nao importa a ordem dos plots
for i in range (0,12):
    im=ax5[i].pcolor(lon,lat,np.mean(evapCum[i:283:12, :, :], axis = 0),norm=mynorm)
    ax5[i].title.set_text(mes[i])    
    
# por estacao
# de um jeito burro pq nao sei a sintax
a=np.arange(start=11,stop=283,step=12)
b=np.arange(start=0,stop=283,step=12)
c=np.arange(start=1,stop=283,step=12)
ver=np.sort(np.concatenate((a,b,c),axis=0))
del a, b ,c
a=np.arange(start=2,stop=283,step=12)
b=np.arange(start=3,stop=283,step=12)
c=np.arange(start=4,stop=283,step=12)
out=np.sort(np.concatenate((a,b,c),axis=0))
del a, b ,c
a=np.arange(start=5,stop=283,step=12)
b=np.arange(start=6,stop=283,step=12)
c=np.arange(start=7,stop=283,step=12)
inv=np.sort(np.concatenate((a,b,c),axis=0))
del a, b ,c
a=np.arange(start=8,stop=283,step=12)
b=np.arange(start=9,stop=283,step=12)
c=np.arange(start=10,stop=283,step=12)
pri=np.sort(np.concatenate((a,b,c),axis=0))
del a, b ,c
estac = ['outono','inverno','primavera','verao']
# out=evapCum.reshape(2,5)[::12].ravel()
mynorm = plt.Normalize(vmin=0, vmax=180)

fig,ax = plt.subplots(2,2,sharex=True) 
ax = ax.flatten() # nao importa a ordem dos plots
# for i in range (0,3): # nao funcionou
#     im = ax[i].pcolor(lon,lat,np.mean(evapCum[out, :, :], axis = 0),norm=mynorm)
#     ax[i].title.set_text(estac[i])
#     fig.colorbar(im, ax=ax[i])
im1 = ax[0].pcolor(lon,lat,np.mean(evapCum[out, :, :], axis = 0),norm=mynorm)
ax[0].title.set_text(estac[0])
fig.colorbar(im1, ax=ax[0])
im2 = ax[1].pcolor(lon,lat,np.mean(evapCum[inv, :, :], axis = 0),norm=mynorm)
ax[1].title.set_text(estac[1])
fig.colorbar(im2, ax=ax[1])
im3 = ax[2].pcolor(lon,lat,np.mean(evapCum[pri, :, :], axis = 0),norm=mynorm)
ax[2].title.set_text(estac[2])
fig.colorbar(im3, ax=ax[2])
im4 = ax[3].pcolor(lon,lat,np.mean(evapCum[ver, :, :], axis = 0),norm=mynorm)
ax[3].title.set_text(estac[3])
fig.colorbar(im4, ax=ax[3])

## Variabilidade temporal (EM UM PIXEL)
# encontrar lat lon mais prox 
def find_nearest(array, values):
    idx = (np.abs(array-values)).argmin()
    return idx
yi=find_nearest(lat, lai)
xi=find_nearest(lon, loi)
#lon[0,17] -50,375 lat[7,0] -27,875

# dfs for descriptive stats on timeseries of choosen point
df_evap = pd.DataFrame(evapCum[:,yi,xi])
df_alb = pd.DataFrame(alb[:,yi,xi])
df_tmp = pd.DataFrame(tmp[:,yi,xi])
df_tveg = pd.DataFrame(tveg[:,yi,xi])

fig5, ax5 = plt.subplots()
ax5.hist(df_evap)
plt.figtext(0.2,0.5, df_evap.describe().to_string())
plt.figtext(0.7,0.75, df_evap.describe().loc[['mean','std']].to_string())
plt.title('Evapotranspiracao')

fig6, ax6 = plt.subplots()
ax6.hist(df_alb)
plt.figtext(0.2,0.5, df_alb.describe().to_string())
plt.figtext(0.7,0.75, df_alb.describe().loc[['mean','std']].to_string())
plt.title('Albedo')

fig7, ax7 = plt.subplots()
ax7.hist(df_tveg)
plt.figtext(0.2,0.5, df_tveg.describe().to_string())
plt.figtext(0.7,0.75, df_tveg.describe().loc[['mean','std']].to_string())
plt.title('Transpiracao Veg')

fig8, ax8 = plt.subplots()
ax8.hist(df_tmp)
plt.figtext(0.2,0.5, df_tmp.describe().to_string())
plt.figtext(0.7,0.75, df_tmp.describe().loc[['mean','std']].to_string())
plt.title('Temperatura do ar')

# variacao no tempo em lages
fig9,ax9 = plt.subplots()
ax9.plot(evapCum[:,yi,xi])
plt.title('Evapotranspiracao no ponto escolhido')

# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(evapCum[:,yi,xi])
print(result)
# decomposicao
components = seasonal_decompose(evapCum[:,yi,xi], model='aditive', period=12) 
fig10 = components.plot()
fig10.suptitle('Evapotranspiracao')

# e a transpiracao da veg?
# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(tveg[:,yi,xi])
print(result)
# decomposicao
components = seasonal_decompose(tveg[:,yi,xi], model='aditive', period=12) 
fig11 = components.plot()
fig11.suptitle('Transpiracao')

# e o albedo reflete as mudanças de uso do solo?
# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(alb[:,yi,xi])
print(result)
# decomposicao
components = seasonal_decompose(alb[:,yi,xi], model='aditive', period=12) 
fig11 = components.plot()
fig11.suptitle('Albedo')

# e a temperatura?
# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(tmp[:,yi,xi])
print(result)
# decomposicao
components = seasonal_decompose(tmp[:,yi,xi], model='aditive', period=12) 
fig11 = components.plot()
fig11.suptitle('Temperatura')


# fazer mann-kendall para toda a area
# baixar dados em area maior, fazer mascara para bacias??