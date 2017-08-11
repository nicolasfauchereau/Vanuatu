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

def get_CRW_SST_raw(day, ll_lat=-21., ur_lat=-12., ll_lon=165., ur_lon=172., base_url = "ftp://ftp.star.nesdis.noaa.gov/pub/sod/mecb/crw/data/5km/v3/nc/v1/daily/sst", opath = "/Users/nicolasf/data/SST/CRW/Vanuatu"):
    infile = "{0:%Y}/b5km_sst_{0:%Y%m%d}.nc".format(day)
    outfile = infile.split('/')[-1]
    cmd = "curl --silent {}/{} -o {}/{}".format(base_url, infile, opath, outfile)

    try:
        r = call(cmd, shell=True)
        print(r)

        dset = xr.open_dataset(os.path.join(opath,outfile))

        sub = dset.sel(lon=slice(ll_lon, ur_lon), lat=slice(ur_lat, ll_lat))

        sub = sub.squeeze()

        print( "filename {}".format( os.path.join( opath, "sub_" + outfile) ) )

        sub.to_netcdf(os.path.join(opath, "sub_" + outfile))

        sub.close()

        dset.close()

        os.remove(os.path.join(opath,outfile))
    except:
        pass

def get_CRW_SST_anomaly(day, ll_lat=-21., ur_lat=-12., ll_lon=165., ur_lon=172., base_url = "ftp://ftp.star.nesdis.noaa.gov/pub/sod/mecb/crw/data/5km/v3/nc/v1/daily/ssta", opath = "/Users/nicolasf/data/SST/CRW/Vanuatu"):
    infile = "{0:%Y}/b5km_ssta_{0:%Y%m%d}.nc".format(day)
    outfile = infile.split('/')[-1]
    cmd = "curl --silent {}/{} -o {}/{}".format(base_url, infile, opath, outfile)

    try:
        r = call(cmd, shell=True)
        print(r)

        dset = xr.open_dataset(os.path.join(opath,outfile))

        sub = dset.sel(lon=slice(ll_lon, ur_lon), lat=slice(ur_lat, ll_lat))

        sub = sub.squeeze()

        print( "filename {}".format( os.path.join( opath, "sub_" + outfile) ) )

        sub.to_netcdf(os.path.join(opath, "sub_" + outfile))

        sub.close()

        dset.close()

        os.remove(os.path.join(opath,outfile))
    except:
        pass


# ### today's date in UTC

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


get_CRW_SST_anomaly(last_available, **domain)

get_CRW_SST_raw(last_available, **domain)


opath = '/Users/nicolasf/data/SST/CRW/Vanuatu/'


lfiles_last_7_days_a = []
lfiles_last_7_days = []

for date in last_7_days:
    lfiles_last_7_days_a.append(os.path.join(opath, 'sub_b5km_ssta_{:%Y%m%d}.nc'.format(date)))
    lfiles_last_7_days.append(os.path.join(opath, 'sub_b5km_sst_{:%Y%m%d}.nc'.format(date)))

lfiles_last_30_days_a = []
lfiles_last_30_days = []

for date in last_30_days:
    lfiles_last_30_days_a.append(os.path.join(opath, 'sub_b5km_ssta_{:%Y%m%d}.nc'.format(date)))
    lfiles_last_30_days.append(os.path.join(opath, 'sub_b5km_sst_{:%Y%m%d}.nc'.format(date)))

for ndays in [1, 7, 30]:

    if ndays == 1:
        dseta = xr.open_dataset(os.path.join(opath, 'sub_b5km_ssta_{:%Y%m%d}.nc'.format(last_available)))

    l = []
    if ndays == 7:
        for f in lfiles_last_7_days_a:
            try:
                dset = xr.open_dataset(f)
                l.append(dset)
            except:
                pass
        dseta = xr.concat(l, dim='time')

    elif ndays == 30:
        for f in lfiles_last_30_days_a:
            try:
                dset = xr.open_dataset(f)
                l.append(dset)
            except:
                pass
        dseta = xr.concat(l, dim='time')


    if ndays == 1:
        dset = xr.open_dataset(os.path.join(opath, 'sub_b5km_sst_{:%Y%m%d}.nc'.format(last_available)))

    l = []
    if ndays == 7:
        for f in lfiles_last_7_days:
            try:
                dset = xr.open_dataset(f)
                l.append(dset)
            except:
                pass
        dset = xr.concat(l, dim='time')

    elif ndays == 30:
        for f in lfiles_last_30_days:
            try:
                dset = xr.open_dataset(f)
                l.append(dset)
            except:
                pass
        dset = xr.concat(l, dim='time')


    if ndays != 1:
        dset = dset.mean('time')
        dseta = dseta.mean('time')


    lat = dset.lat
    lon = dset.lon

    if os.path.exists( "/Users/nicolasf/operational/Vanuatu/code/basemap_pickle_SST.pickle" ):
        m = pickle.load( open( "/Users/nicolasf/operational/Vanuatu/code/basemap_pickle_SST.pickle", "rb" ) )
    else:
        m = bm(projection='cyl',llcrnrlat=lat.data.min(),urcrnrlat=lat.data.max(), llcrnrlon=lon.data.min(),urcrnrlon=lon.data.max(),            lat_ts=0,resolution='f')
        pickle.dump( m, open( "/Users/nicolasf/operational/Vanuatu/code/basemap_pickle_SST.pickle", "wb" ) )

    lons, lats = np.meshgrid(lon, lat)

    jet = plt.get_cmap('jet')

    f = plot_field_contourf(m, dset['CRW_SST'], lats, lons, 22.5, 30, 0.25, grid=True, title='CRW SST, last {} days to {:%Y-%m-%d}\nData made available by NOAA Coral Reef Watch'.format(ndays, last_available), cmap=jet)


    f.savefig('/Users/nicolasf/operational/Vanuatu/figures/CRW_SST_last_{}days.png'.format(ndays), dpi=200)

    f = plot_field_contourf(m, dseta['CRW_SSTANOMALY'], lats, lons, -1.5, 1.5, 0.05, grid=True, title='CRW SST anomaly, last {} days to {:%Y-%m-%d}\nData made available by NOAA Coral Reef Watch'.format(ndays, last_available), cmap=cm.balance)

    f.savefig('/Users/nicolasf/operational/Vanuatu/figures/CRW_SST_anomaly_last_{}days.png'.format(ndays), dpi=200)

    dset.close()

    dseta.close()
