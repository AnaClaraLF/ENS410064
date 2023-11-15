# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:18:43 2023

@author: 55489
"""

# pacotes
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr


# caminho para meu arquivo com conc de o3
path = r"D:\ENS410064\Dados\brutos\earth\GLDAS2 GLDAS_NOAH025_3H_v2.1 Evap_tavg_-49.9570_-27.7769_20170101_20171231.nc"
path2 = r"D:\ENS410064\Dados\brutos\earth\GLDAS_CLSM025_D.A19480101.020.nc4.SUB.nc4"



# abrir arquivo
data = nc.Dataset(path)
data2 = nc.Dataset(path2)


# extraindo dados 
lon = data2['lon'][:]
lat = data2['lat'][:]
rain = data2['Rainf_tavg'][:]

#no espaco
fig,ax = plt.subplots()
ax.pcolor(lon,lat,rain[0, :, :])
