#!/Users/nicolasf/anaconda/envs/IOOS3/bin/python

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap as bm
from mpl_toolkits.basemap import interp
import xarray as xr
from glob import glob

from datetime import datetime, timedelta
from dateutil.parser import parse as dparse
import calendar

from cmocean import cm

import pickle
from subprocess import call

"""
functions definitions
"""

def NN_interp(mat):
    import numpy as np
    from numpy import ma
    from scipy.spatial import cKDTree
    mat = ma.masked_array(mat, np.isnan(mat))
    a = mat.copy()
    a._sharedmask=False
    x,y=np.mgrid[0:a.shape[0],0:a.shape[1]]
    xygood = np.array((x[~a.mask],y[~a.mask])).T
    xybad = np.array((x[a.mask],y[a.mask])).T
    a[a.mask] = a[~a.mask][cKDTree(xygood).query(xybad)[1]]
    a = np.array(a)
    return a

def plot_field_pcolor(m, X, lats, lons, vmin, vmax, step, cmap=plt.get_cmap('RdBu_r'), ax=False, title=False, grid=False, cbar_label = u"\u00b0" + "C"):
    if not ax:
        f, ax = plt.subplots(figsize=(8, (X.shape[0] / float(X.shape[1])) * 8))
    m.ax = ax
#     im = m.contourf(lons, lats, X, np.arange(vmin, vmax+step, step), \
#                     latlon=True, cmap=cmap, extend='both', ax=ax)
    im = m.pcolormesh(lons, lats, X, vmin=vmin, vmax=vmax, latlon=True, cmap=cmap, ax=ax)

    m.drawcoastlines()
    m.fillcontinents(color='0.8')
    if grid:
        m.drawmeridians(np.arange(0, 360, 2.5), labels=[0,0,0,1])
        m.drawparallels(np.arange(-80, 80, 2.5), labels=[1,0,0,0])
    cb = m.colorbar(im)
    [l.set_fontsize(14) for l in cb.ax.yaxis.get_ticklabels()]
    cb.set_label(u"\u00b0" + "C", fontsize=14)
    if title:
        ax.set_title(title, fontsize=15)


def plot_field_contourf(m, X, lats, lons, vmin, vmax, step, cmap=plt.get_cmap('RdBu_r'), title=False, grid=False, cbar_label = u"\u00b0" + "C"):

    f, ax = plt.subplots(figsize=(8, (X.shape[0] / float(X.shape[1])) * 8))
    f.subplots_adjust(right=0.85)
    m.ax = ax
    im = m.contourf(lons, lats, X, np.arange(vmin, vmax+step, step), latlon=True, cmap=cmap, extend='both', ax=ax)

    m.drawcoastlines()
    m.fillcontinents(color='0.8')
    if grid:
        m.drawmeridians(np.arange(0, 360, 2.5), labels=[0,0,0,1], fontsize=14)
        m.drawparallels(np.arange(-80, 80, 2.5), labels=[1,0,0,0], fontsize=14)
    cb = m.colorbar(im)
    [l.set_fontsize(14) for l in cb.ax.yaxis.get_ticklabels()]
    cb.set_label(u"\u00b0" + "C", fontsize=14)
    if title:
        ax.set_title(title, fontsize=15)
    return f


def get_OISST_data(day, ll_lat=-21., ur_lat=-12., ll_lon=165., ur_lon=172., base_url = "https://www.ncei.noaa.gov/thredds", opath = "/Users/nicolasf/data/SST/NOAA_hires_1981_present/Vanuatu/"):

    url_https_final = 'https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/AVHRR/{0:%Y%m}/avhrr-only-v2.{0:%Y%m%d}.nc'.format(day)

    url_https_prelim = 'https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/AVHRR/{0:%Y%m}/avhrr-only-v2.{0:%Y%m%d}_preliminary.nc'.format(day)

    outfile = url_https_final.split('/')[-1]


    # curl needs to be called with the `-k` flag to access unsecure https connection
    cmd_f = "curl -k {} -o {}/{}".format(url_https_final, opath, outfile)

    cmd_p = "curl -k {} -o {}/{}".format(url_https_prelim, opath, outfile)

    # try and see if final file is available
    r = call(cmd_f, shell=True)

    stat_info = os.stat(os.path.join(opath,outfile))

    # if the file is too small, the file wasnt actually available remove it and get the prelim file
    if stat_info.st_size < 800:
        os.remove(os.path.join(opath,outfile))
        r = call(cmd_p, shell=True)

    stat_info = os.stat(os.path.join(opath,outfile))

    print("call status {}, filesize {}".format(r, stat_info.st_size))

    dset = xr.open_dataset(os.path.join(opath,outfile))

    sub = dset.sel(lon=slice(ll_lon, ur_lon), lat=slice(ll_lat, ur_lat))[['anom','sst']]

    sub = sub.squeeze()

    print( "filename {}".format( os.path.join( opath, "sub_" + outfile) ) )

    sub.to_netcdf(os.path.join(opath, "sub_" + outfile))

    sub.close()

    dset.close()

    os.remove(os.path.join(opath,outfile))


"""
get the date of today (must be run before 12)
"""

today = datetime.utcnow()

delay = 1

# ### last day available

last_available = today - timedelta(days=delay)

last_7_days = pd.date_range(start=last_available - timedelta(days=6), end=last_available, freq='1D')

last_30_days = pd.date_range(start=last_available - timedelta(days=29), end=last_available, freq='1D')

domain = {}
domain['ll_lat'] = -21
domain['ur_lat'] = -12
domain['ll_lon'] = 165.
domain['ur_lon'] = 172.


get_OISST_data(last_available, **domain)

opath = "/Users/nicolasf/data/SST/NOAA_hires_1981_present/Vanuatu/"

lfiles_last_7_days = []
for date in last_7_days:
    lfiles_last_7_days.append(os.path.join(opath, 'sub_avhrr-only-v2.{:%Y%m%d}.nc'.format(date)))

lfiles_last_30_days = []
for date in last_30_days:
    lfiles_last_30_days.append(os.path.join(opath, 'sub_avhrr-only-v2.{:%Y%m%d}.nc'.format(date)))

for ndays in [1, 7, 30]:

    if ndays == 1:
        dset = xr.open_dataset(os.path.join(opath, 'sub_avhrr-only-v2.{:%Y%m%d}.nc'.format(last_available)))

    elif ndays == 7:
        dset = xr.open_mfdataset(lfiles_last_7_days, concat_dim='time')

    elif ndays == 30:
        dset = xr.open_mfdataset(lfiles_last_30_days, concat_dim='time')

    if ndays != 1:
        dset = dset.mean('time')

    lat = dset.lat
    lon = dset.lon

    if os.path.exists( "/Users/nicolasf/operational/Vanuatu/code/basemap_pickle_SST.pickle" ):
        m = pickle.load( open( "/Users/nicolasf/operational/Vanuatu/code/basemap_pickle_SST.pickle", "rb" ) )
    else:
        m = bm(projection='cyl',llcrnrlat=lat.data.min(),urcrnrlat=lat.data.max(), llcrnrlon=lon.data.min(),urcrnrlon=lon.data.max(),            lat_ts=0,resolution='f')
        pickle.dump( m, open( "/Users/nicolasf/operational/Vanuatu/code/basemap_pickle_SST.pickle", "wb" ) )


    lons, lats = np.meshgrid(lon, lat)

    anoms  = NN_interp(dset['anom'].data)
    raw = NN_interp(dset['sst'].data)


    new_lon = np.linspace(domain['ll_lon'], domain['ur_lon'], lon.shape[0]*2, endpoint=True)
    new_lat = np.linspace(domain['ll_lat'], domain['ur_lat'], lat.shape[0]*2, endpoint=True)

    new_lons, new_lats = np.meshgrid(new_lon, new_lat)

    anoms_interp = interp(anoms, lon.data, lat.data, new_lons, new_lats)
    raw_interp = interp(raw, lon.data, lat.data, new_lons, new_lats)

    f = plot_field_contourf(m, raw_interp, new_lats, new_lons, 22.5, 30, 0.15, grid=True, title='NOAA OISST last {} days to {:%Y-%m-%d}'.format(ndays, last_available), cmap=plt.get_cmap('jet'))

    f.savefig('/Users/nicolasf/operational/Vanuatu/figures/NOAA_OISST_last_{}days.png'.format(ndays), dpi=200)

    f = plot_field_contourf(m, anoms_interp, new_lats, new_lons, -1.5, 1.5, 0.05, grid=True, title='NOAA OISST last {} days anomalies to {:%Y-%m-%d}'.format(ndays, last_available), cmap=cm.balance)

    f.savefig('/Users/nicolasf/operational/Vanuatu/figures/NOAA_OISST_last_{}days_anomalies.png'.format(ndays), dpi=200)

    dset.close()
