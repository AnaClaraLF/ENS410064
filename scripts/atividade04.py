# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:18:43 2023

@author: 55489
"""
## dados de -entrada-
# ponto a ser investigado - Lages~ -27.81, -50.31 - Coxilha -28,20 -50,33
lai = -28.20
loi = -50.33
t0 = '2000-01-01' # quando inicia a serie temporal
## dados de -entrada-

# pacotes
#import os
#import dask
import datetime
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

# extraindo dados 
lon = data['lon'][:]
lat = data['lat'][:]
evap = data['Evap_tavg'][:]
alb = data['Albedo_inst'][:]
tmp = data['Tair_f_inst'][:]
tveg = data['Tveg_tavg'][:]
tempo = data['time'] # dias desde 01-01-2000
dia = data['time_bnds'][:] #intervalo dias cada mes
diasm = dia[:,1]-dia[:,0] # quantos dias cada mes
# temperatura em celsius??
# acumulado da evap mensal, kg de h2o/m² = mm de h2o
evapCum = ma.empty([tempo.size,lat.size,lon.size])
for i in range(len(diasm)):
    evapCum[i,:,:] = evap[i,:,:].__mul__(10800*8*diasm[i]) # em mm/mes

# criando vetor temporal de datas
dtime = np.arange(np.datetime64(t0), np.datetime64(t0) + np.timedelta64(np.int64(dia[tempo.size-1,1]),'D') , dtype='datetime64[M]')

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
est = [ver,out,inv,pri]
estac = ['verao','outono','inverno','primavera']
mynorm = plt.Normalize(vmin=0, vmax=180)

fig,ax = plt.subplots(2,2,sharex=True) 
ax = ax.flatten() # nao importa a ordem dos plots
for i in range (0,4): # nao funcionou
    im = ax[i].pcolor(lon,lat,np.mean(evapCum[est[i], :, :], axis = 0),norm=mynorm)
    ax[i].title.set_text(estac[i])
    fig.colorbar(im, ax=ax[i])    

## Variabilidade temporal (EM UM PIXEL)
# encontrar lat lon mais prox do ponto escolhido la no comeco do script
def find_nearest(array, values):
    idx = (np.abs(array-values)).argmin()
    return idx
yi=find_nearest(lat, lai)
xi=find_nearest(lon, loi)
#lon[0,17] -50,375 lat[7,0] -27,875

# dfs for descriptive stats on timeseries of choosen point
df_evap = pd.DataFrame(evapCum[:,yi,xi])
df_evap['datetime'] = dtime # set index datetime
df_evap = df_evap.set_index(df_evap['datetime']) # set index datetime
df_evap.drop('datetime',axis=1, inplace=True)# drop column

df_alb = pd.DataFrame(alb[:,yi,xi])
df_alb['datetime'] = dtime # set index datetime
df_alb = df_alb.set_index(df_alb['datetime']) # set index datetime
df_alb.drop('datetime',axis=1, inplace=True)# drop column

df_tmp = pd.DataFrame(tmp[:,yi,xi])
df_tmp['datetime'] = dtime # set index datetime
df_tmp = df_tmp.set_index(df_tmp['datetime']) # set index datetime
df_tmp.drop('datetime',axis=1, inplace=True)# drop column

df_tveg = pd.DataFrame(tveg[:,yi,xi])
df_tveg['datetime'] = dtime # set index datetime
df_tveg = df_tveg.set_index(df_tveg['datetime']) # set index datetime
df_tveg.drop('datetime',axis=1, inplace=True)# drop column

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
ax9.plot(df_evap)#evapCum[:,yi,xi])
plt.title('Evapotranspiracao no ponto escolhido')

# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(df_evap)#evapCum[:,yi,xi])
print(result)
# decomposicao
components = seasonal_decompose(df_evap, model='aditive', period=12)#evapCum[:,yi,xi] 
fig10 = components.plot()
fig10.suptitle('Evapotranspiracao')

# e a transpiracao da veg?
# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(df_tveg)
print(result)
# decomposicao
components = seasonal_decompose(df_tveg, model='aditive', period=12) 
fig11 = components.plot()
fig11.suptitle('Transpiracao')

# e o albedo reflete as mudanças de uso do solo?
# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(df_alb)
print(result)
# decomposicao
components = seasonal_decompose(df_alb, model='aditive', period=12) 
fig11 = components.plot()
fig11.suptitle('Albedo')

# e a temperatura?
# verificar se há tendencia e sazonalidade
# mann-kendall
result = mk.original_test(df_tmp)
print(result)
# decomposicao
components = seasonal_decompose(df_tmp, model='aditive', period=12) 
fig11 = components.plot()
fig11.suptitle('Temperatura')


# fazer mann-kendall para toda a area
# iterar?
output = []
for i in np.arange(len(evapCum[0,:,0])):
    for j in np.arange(len(evapCum[0,0,:])):
        trend = mk.original_test(evapCum[:,i,j]).trend
        output.append(trend)

output = np.copy(output).reshape(len(lat),len(lon))
trends = ['decreasing','no trend','increasing'] # to replace

output[output == trends[0]]=int(-1)
output[output == trends[1]]=int(0)
output[output == trends[2]]=int(1)
output = ma.masked_array(output,evapCum[0,:,:].mask)
output.astype(float)
# plot
fig12,ax12 = plt.subplots()
ax12.pcolor(lon, lat, output)
plt.title('trend')

# baixar dados em area maior, fazer mascara para bacias??
# trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(evapCum[:,i,j])






