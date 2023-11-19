# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:53:38 2023

@author: 55489
"""
# pacotes
import xarray as xr
import dask

dask.config.set({"array.slicing.split_large_chunks": False})
ds = xr.open_mfdataset('D:/ENS410064/Dados/brutos/earth/*.nc4', combine='nested', concat_dim="time")
ds.to_netcdf('D:/ENS410064/Dados/brutos/earth/all-nc-files.nc4')

# import nctoolkit as nct
# concatenar ncs
# caminho para os arquivos
#path = r'D:\ENS410064\Dados\brutos\earth'
#ds = nct.open_data("path/*.nc")
# ds1 = nct.open_data("D:/ENS410064/Dados/brutos/earth/GLDAS_NOAH025_M.A202301.021.nc4.SUB.nc4")
# ds2 = nct.open_data(r"D:\ENS410064\Dados\brutos\earth\GLDAS_NOAH025_M.A202302.021.nc4.SUB.nc4")
# merge the files
#ds1.append(ds2)
#ds1.merge()
# save the files as a netcdf file
#ds1.to_nc("path/merged.nc")

# # caminho para os arquivos
# path = r'D:\ENS410064\Dados\brutos\earth'
# # lista dos nomes completos dos arquivos nc
# lsta = []
# for file in os.listdir(path):
#     if file.endswith(".nc4"):
#         print(os.path.join(path, file))
#         lsta = lsta + [os.path.join(path,file)]