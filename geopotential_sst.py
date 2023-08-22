# %%
import xarray as xr
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from scipy.signal import detrend

fn_dir = '/data0/jlin/strat/ncep'

fn_air = 'air.mon.mean.nc'
fn_geo = 'hgt.mon.mean.nc'
fn_sst = 'sst.mnmean.nc'

ds_air = xr.open_dataset('%s/%s' % (fn_dir, fn_air))
ds_geo = xr.open_dataset('%s/%s' % (fn_dir, fn_geo))
ds_sst = xr.open_dataset('%s/%s' % (fn_dir, fn_sst)).sel(time = slice(ds_air['time'][0], ds_air['time'][-1]))

temp = ds_air['air'].data
sst = ds_sst['sst'].data
geo = ds_geo['hgt'].data

lon = ds_air['lon'].data
lat = ds_air['lat'].data
lvl = ds_air['level'].data

# %% Create anomalies by deseasonalizing
ds_dts = dts = np.array([dt.datetime.strptime(str(x), '%Y-%m') for x in np.asarray(ds_air['time']).astype('datetime64[M]')])
ds_months = np.array([x.month for x in ds_dts])
months = np.arange(1, 13, 1)
temp_monthly = np.zeros((12,) + ds_air['air'].shape[1:])
sst_monthly = np.zeros((12,) + ds_sst['sst'].shape[1:])
geo_monthly = np.zeros((12,) + ds_geo['hgt'].shape[1:])
for m in months:
    print(m)
    month_mask = ds_months == m
    temp_monthly[m-1, :, :, :] = np.nanmean(temp[month_mask, :, :, :], axis = 0)
    geo_monthly[m-1, :, :, :] = np.nanmean(geo[month_mask, :, :, :], axis = 0)
    sst_monthly[m-1, :, :] = np.nanmean(sst[month_mask, :, :], axis = 0)

# %% Calculate anomalies of all variables with seasonal cycle removed.
temp_anom = np.zeros(temp.shape)
sst_anom = np.zeros(sst.shape)
geo_anom = np.zeros(geo.shape)
for m in months:
    print(m)
    month_mask = ds_months == m
    temp_anom[month_mask, :, :, :] = temp[month_mask, :, :, :] - np.tile(temp_monthly[m-1, :, :, :], (np.sum(month_mask), 1, 1, 1))
    geo_anom[month_mask, :, :, :] = geo[month_mask, :, :, :] - np.tile(geo_monthly[m-1, :, :, :], (np.sum(month_mask), 1, 1, 1))
    sst_anom[month_mask, :, :] = sst[month_mask, :, :] - np.tile(sst_monthly[m-1, :, :], (np.sum(month_mask), 1, 1))

# %% Define warm SST and cold SST tropical events.
sst_tropics = detrend(np.nanmean(sst_anom, axis = (1, 2)))
sst_warm = sst_tropics >= (1*np.std(sst_tropics))
sst_cold = sst_tropics <= (-1*np.std(sst_tropics))
sst_neutral = np.abs(sst_tropics) <= 1*np.std(sst_tropics)

geo_warm = np.nanmean(geo_anom[sst_warm, :, :, :], axis = (0, 2, 3))
geo_cold = np.nanmean(geo_anom[sst_cold, :, :, :], axis = (0, 2, 3))
geo_neutral = np.nanmean(geo_anom[sst_neutral, :, :, :], axis = (0, 2, 3))

temp_warm = np.nanmean(temp_anom[sst_warm, :, :, :], axis = (0, 2, 3))
temp_cold = np.nanmean(temp_anom[sst_cold, :, :, :], axis = (0, 2, 3))
temp_neutral = np.nanmean(temp_anom[sst_neutral, :, :, :], axis = (0, 2, 3))

# % Plot warm and cold events.
plt.figure(); plt.plot(sst_tropics, 'k');
plt.scatter(np.array(range(len(sst_tropics)))[sst_warm], sst_tropics[sst_warm], color='r')
plt.scatter(np.array(range(len(sst_tropics)))[sst_cold], sst_tropics[sst_cold], color='b')

# %% Plot geopotential anomaly profile for warm SST and cold SST months
lvl_labels = [1000, 700, 500, 350, 250, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(figsize=(16, 8), ncols = 2);

warm_idxs = np.array([x for x in np.argwhere(sst_warm).flatten() if x < sst_warm.shape[0]])
cold_idxs = np.array([x for x in np.argwhere(sst_cold).flatten() if x < sst_cold.shape[0]])

anom_total = np.nanmean(sst_tropics[warm_idxs]) - np.nanmean(sst_tropics[cold_idxs])
l_weight = np.nanmean(sst_tropics[warm_idxs]) / anom_total
geo_warm = np.nanmean(geo_anom[warm_idxs, :, :, :][:, :, np.abs(lat) <= 15, :], axis = (0, 2, 3))
geo_cold = np.nanmean(geo_anom[cold_idxs, :, :, :][:, :, np.abs(lat) <= 15, :], axis = (0, 2, 3))
geo_avg = geo_warm * l_weight - geo_cold * (1 - l_weight)
temp_warm = np.nanmean(temp_anom[warm_idxs, :, :, :][:, :, np.abs(lat) <= 15, :], axis = (0, 2, 3))
temp_cold = np.nanmean(temp_anom[cold_idxs, :, :, :][:, :, np.abs(lat) <= 15, :], axis = (0, 2, 3))
temp_avg = temp_warm * l_weight - temp_cold * (1 - l_weight)

axs[0].plot(geo_warm, np.log(lvl), 'r-');
axs[0].plot(geo_cold, np.log(lvl), 'b-');
axs[0].plot(geo_avg, np.log(lvl), 'k-')

axs[1].plot(temp_warm, np.log(lvl), 'r-');
axs[1].plot(temp_cold, np.log(lvl), 'b-');
axs[1].plot(temp_avg, np.log(lvl), 'k-')

for ax in axs:
    ax.grid();
    ax.set_yticks(np.log(lvl_labels))
    ax.set_yticklabels(lvl_labels)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_ylim([np.log(10), np.log(1000)])
    ax.invert_yaxis()
axs[0].set_xlabel('Geopotential ($m^s/s^2$)')
axs[1].set_xlabel('Temperature (K)')

# %% Running mean
geo_tropics = np.nanmean(geo_anom, axis = (2, 3))
geo_cum = np.cumsum(geo_tropics, axis = 0)
geo_cum[12:] = geo_cum[12:] - geo_cum[:-12]
geo_yrly = geo_cum[12 - 1:] / 12