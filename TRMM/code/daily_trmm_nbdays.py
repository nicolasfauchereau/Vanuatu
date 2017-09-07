#!/Users/nicolasf/anaconda/envs/IOOS3/bin/python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap as bm
import xarray as xr
from numpy import ma

from datetime import datetime, timedelta

import cartopy.crs as ccrs
from cartopy import feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


sys.path.append('/Users/nicolasf/pythonlibs/resource_converter/')
from regrid import regrid

lag = 2

today = datetime.utcnow() - timedelta(days=lag)

today_local = today + timedelta(hours=11)

last_90_days = pd.date_range(start=today - timedelta(days=90), end=today, freq='1D')

last_30_days = pd.date_range(start=today - timedelta(days=30), end=today, freq='1D')

root_dpath = '/Users/nicolasf/data'

dpath = os.path.join(root_dpath, 'TRMM/daily')

# list of files for the past 90 days

lfiles_90_days = []
for d in last_90_days:
    fname  = os.path.join(dpath, "3B42RT_daily.{:%Y.%m.%d}.nc".format(d))
    lfiles_90_days.append(fname)

# list of files for the past 30 days

lfiles_30_days = []
for d in last_30_days:
    fname  = os.path.join(dpath, "3B42RT_daily.{:%Y.%m.%d}.nc".format(d))
    lfiles_30_days.append(fname)

dset_90_days = xr.open_mfdataset(lfiles_90_days)
dset_30_days = xr.open_mfdataset(lfiles_30_days)

"""
functions definitions
"""
def get_last_dry_sequence(x, thresh=0):
    x[x>thresh] = 1
    x = x[::-1]
    nl = np.where(x == 1)[0]
    if nl.size == 0:
        n = 90
    else:
        n = nl[0]
    return n

def round_down(num, divisor):
    return num - (num%divisor)

"""
functions definitions
"""

region = 'Vanuatu'

if region == 'Vanuatu':
    domain = {'latmin':-21, 'lonmin':165., 'latmax':-12, 'lonmax':172.}
elif region == 'Samoa':
    domain = {'latmin':-16, 'lonmin':(360. - 175.5), 'latmax':-11.5, 'lonmax':(360. - 167.25)}
else:
    print('need a region: Vanuatu or Samoa')

dset_90_days = dset_90_days.sel(lon=slice(domain['lonmin'],domain['lonmax']), lat=slice(domain['latmin'], domain['latmax']))
dset_30_days = dset_30_days.sel(lon=slice(domain['lonmin'],domain['lonmax']), lat=slice(domain['latmin'], domain['latmax']))


lat = dset_30_days.lat.data
lon = dset_30_days.lon.data

arr_nb = np.empty((3, len(lat), len(lon)))
for i, n in enumerate([0, 1, 5]):
    nbdays = np.apply_along_axis(get_last_dry_sequence, 0, dset_90_days['trmm'].data, thresh=n)
    arr_nb[i,:,:] = nbdays

dset_nb_days = {}
dset_nb_days['lat'] = (('lat'), dset_90_days.lat)
dset_nb_days['lon'] = (('lon'), dset_90_days.lon)
dset_nb_days['thresh'] = ('thresh', np.array([0, 1, 5]))
dset_nb_days['nbdays'] = (('thresh','lat','lon'), arr_nb)

dset_nb_days = xr.Dataset(dset_nb_days)

# dset_nb_days.sel(thresh=5)['nbdays'].plot()

new_res = 5

new_lon = np.linspace(lon[0], lon[-1], len(lon) * new_res, endpoint=True)

new_lat = np.linspace(lat[0], lat[-1], len(lat) * new_res, endpoint=True)

dset_30_days.load()

new_lon, new_lat, matm = regrid(dset_30_days.sum('time')['trmm'].data, lon, lat, new_lon, new_lat)

lons, lats = np.meshgrid(new_lon, new_lat)

sys.path.append(os.path.join(os.environ['HOME'], 'pythonlibs'))

import nclcmaps

import nclcmaps as ncm
cmap = ncm.cmap('GMT_drywet')

proj = ccrs.PlateCarree()

f, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=proj))

ax.set_extent([domain['lonmin'], domain['lonmax'], domain['latmin'], domain['latmax']])

ax.add_feature(cfeature.GSHHSFeature(scale='full', levels=[1,2,3,4]))

im = ax.contourf(lons, lats, matm, levels=np.arange(0, round_down(matm.max(), 20), 20), cmap=cmap, extend='max', transform=proj)

c = ax.contour(lons, lats, matm, levels=np.arange(0, round_down(matm.max(), 20), 20), color='k', linewidths=0.5, transform=proj)

cb = plt.colorbar(im, shrink=0.8)

cb.set_label('mm')

gl = ax.gridlines(draw_labels=True, lw=0.2, linestyle=':')

gl.xlabels_top = gl.ylabels_right = False

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

ax.set_title('30 days cumulative rainfall (mm) to {:%Y-%m-%d}'.format(today_local))

f.savefig('../figures/cumulative_rainfall_last30days.png', dpi=200)

cmap = ncm.cmap('GMT_haxby')

for thresh in [0, 1, 5]:

    """
    interpolate
    """

    matnbdays = dset_nb_days.sel(thresh=thresh)['nbdays'].data
    new_lon, new_lat, matnbdays = regrid(matnbdays, lon, lat, new_lon, new_lat)

    """
    set the figure
    """

    f, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=proj))

    ax.set_extent([domain['lonmin'], domain['lonmax'], domain['latmin'], domain['latmax']])

    """
    add the coastline
    """

    ax.add_feature(cfeature.GSHHSFeature(scale='full', levels=[1,2,3,4]), edgecolor='k', linewidth=1)

    """
    add the field
    """

    im = ax.contourf(lons, lats, matnbdays, levels=np.arange(0, round_down(matnbdays.max(), 5) + 5, 5), cmap=cmap, extend='max')

#     c = ax.contour(lons, lats, matnbdays, levels=np.arange(0, round_down(matnbdays.max(), 5) + 5, 5), color='k', linewidths=0.5)

    gl = ax.gridlines(draw_labels=True, lw=0.2, linestyle=':')

    gl.xlabels_top = gl.ylabels_right = False

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    cb = plt.colorbar(im, shrink=0.8)

    cb.set_label('days')
    ax.set_title('number of days since last rainfall > {} mm\nvalid to {:%Y-%m-%d}'.format(thresh, today_local))
    f.savefig('../figures/nbdays_thresh_{}mm.png'.format(thresh), dpi=200)
