# -*- coding: utf-8 -*-

# pacotes
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# caminho para meu arquivo com conc de o3
path = r'D:\ENS410064\Dados\brutos\ufsc\BRAIN_Clipped_CONC_O3_2019_07_02_11_to_2019_12_30_23.nc'

# abrir arquivo
data = nc.Dataset(path)
# extraindo dados de o3
o3 = data['O3'][:]
lat = data['LAT'][:]
lon = data['LON'][:]
tflag = data['TFLAG'][:]

# Variabilidade temporal (em um pixel)
# media em ~lages -27.81, -50.31
# media em ~timbo -26.82 -49.26
# encontrar lat lon mais prox
#lon[0,21] -49.22636 lat[37,0] -26.808065
meantb = np.mean(o3[:,0,21,37],axis = 0)
maxtb = np.max(o3[:,0,21,37],axis = 0)
mintb = np.min(o3[:,0,21,37],axis = 0)

#no tempo
fig,ax = plt.subplots()
ax.plot(o3[:,0,21,37])

# Variabilidade espacial(médias em todo tempo, por estação, por mês)
# todo intervalo
meant = np.mean(o3[:,0,:,:],axis = 0)
maxt = np.max(o3[:,0,:,:],axis = 0)
mint = np.min(o3[:,0,:,:],axis = 0)

#no espaco
# media
fig2,ax2 = plt.subplots()
ax2.pcolor(lon,lat,meant)
# maximas
fig3,ax3 = plt.subplots()
ax3.pcolor(lon,lat,maxt)
# minimas
fig4,ax4 = plt.subplots()
ax4.pcolor(lon,lat,mint)
#outono

#inverno

#primavera

#verao


