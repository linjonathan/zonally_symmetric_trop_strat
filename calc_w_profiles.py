# %%
import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import os

os.chdir('/home/jlin/src/strat')

fn = '/data0/jlin/strat/era5/strat_vertical_velocity_monthly.nc'
ds = xr.open_dataset(fn)
lon = ds['longitude']
lat = ds['latitude']
lvl = ds['level']
omega = ds['w'][:-2, 0, :, np.abs(lat) <= 20, :]

fn_geo = '/data0/jlin/strat/era5/strat_geopotential_monthly.nc'
ds_geo = xr.open_dataset(fn_geo)
geo = ds_geo['z'][:-1, 0, :, np.abs(lat) <= 20, :]
time = ds_geo['time'][:-1]
dts = np.array([dt.datetime.strptime(str(x), '%Y-%m') for x in np.asarray(time).astype('datetime64[M]')])

#fn_pv = '/data0/jlin/strat/era5/pv_monthly.nc'
#ds_pv = xr.open_dataset(fn_pv)
#pv = ds_pv['pv'][:-7, :, np.abs(lat) <= 20, :]

fn_temp = '/data0/jlin/strat/era5/strat_T_monthly.nc'
ds_temp = xr.open_dataset(fn_temp)
temp = ds_temp['t'][:-1, 0, :, np.abs(lat) <= 20, :]

lvl_mat_hPa = np.tile(np.moveaxis(np.tile(lvl, (81, 720, 1)), -1, 0), (509, 1, 1, 1))
rho = np.divide(lvl_mat_hPa * 100, 287.058 * temp)
w = -np.divide(omega, rho * 9.81)

fn_sst = '/data0/jlin/strat/era5/sst_monthly.nc'
ds_sst = xr.open_dataset(fn_sst)
sst = ds_sst['sst'][0:w.shape[0], np.abs(lat) <= 20, :]

w_months = np.array([x.month for x in dts])
months = np.arange(1, 13, 1)
w_monthly = np.zeros((12,) + w.shape[1:])
temp_monthly = np.zeros((12,) + temp.shape[1:])
sst_monthly = np.zeros((12,) + sst.shape[1:])
geo_monthly = np.zeros((12,) + geo.shape[1:])
#pv_monthly = np.zeros((12,) + pv.shape[1:])
for m in months:
    print(m)
    month_mask = w_months == m

    temp_monthly[m-1, :, :, :] = np.nanmean(temp[month_mask, :, :, :], axis = 0)
    #pv_monthly[m-1, :, :, :] = np.nanmean(pv[month_mask, :, :, :], axis = 0)
    geo_monthly[m-1, :, :, :] = np.nanmean(geo[month_mask, :, :, :], axis = 0)
    sst_monthly[m-1, :, :] = np.nanmean(sst[month_mask, :, :], axis = 0)

# %% Calculate anomalies of all variables with seasonal cycle removed.
w_anom = np.zeros(w.shape)
temp_anom = np.zeros(temp.shape)
sst_anom = np.zeros(sst.shape)
geo_anom = np.zeros(geo.shape)
#pv_anom = np.zeros(pv.shape)
for m in months:
    print(m)
    month_mask = w_months == m
    temp_anom[month_mask, :, :, :] = temp[month_mask, :, :, :] - np.tile(temp_monthly[m-1, :, :, :], (np.sum(month_mask), 1, 1, 1))
    #pv_anom[month_mask, :, :, :] = pv[month_mask, :, :, :] - np.tile(pv_monthly[m-1, :, :, :], (np.sum(month_mask), 1, 1, 1))
    geo_anom[month_mask, :, :, :] = geo[month_mask, :, :, :] - np.tile(geo_monthly[m-1, :, :, :], (np.sum(month_mask), 1, 1, 1))
    sst_anom[month_mask, :, :] = sst[month_mask, :, :] - np.tile(sst_monthly[m-1, :, :], (np.sum(month_mask), 1, 1))

# %% Calculate anomalies with the annual mean subtracted out.
geo_anom_annual = geo - np.nanmean(geo, axis = 0)

# %% Running mean, and detrend the temperature and geopotential
from scipy.signal import detrend

geo_tropics = np.nanmean(geo_anom, axis = (2, 3))
geo_cum = np.cumsum(geo_tropics, axis = 0)
geo_cum[12:] = geo_cum[12:] - geo_cum[:-12]
geo_yrly = geo_cum[12 - 1:] / 12

lat_tropics = lat[np.abs(lat) <= 20]
geo_trend = geo_tropics - detrend(geo_tropics, axis = 0)
geo_mat_trend = np.tile(np.expand_dims(geo_trend, [2,3]), (1, 1, lat_tropics.shape[0], lon.shape[0]))
geo_anom_detrend = geo_anom - geo_mat_trend

temp_tropics = np.nanmean(temp_anom, axis = (2, 3))
temp_trend = temp_tropics - detrend(temp_tropics, axis = 0)
temp_mat_trend = np.tile(np.expand_dims(temp_trend, [2,3]), (1, 1, lat_tropics.shape[0], lon.shape[0]))
temp_anom_detrend = temp_anom - temp_mat_trend

# %% Define atmospheric levels
fn_iso = '/data0/jlin/strat/iso_atmosphere.txt'
iso_data = np.loadtxt(fn_iso)
iso_p = iso_data[:, 3]
iso_z = iso_data[:, 5]
lvl_z = np.flip(np.interp(lvl, iso_p, iso_z))

from scipy.ndimage import uniform_filter1d

# %%
LON, LAT = np.meshgrid(lon, lat[np.abs(lat) <= 20])
wp_mask = (LON >= 45) & (LON <= 180) & (LAT <= 20)
ep_mask = (LON <= -80) & (LON >= -180) & (LAT <= 20)
al_mask = (LON >= -60) & (LON <= 0) & (LAT <= 20)

sst_tropics = detrend(np.nanmean(sst_anom, axis = (1, 2)))
sst_wp = detrend(np.nanmean(sst_anom[:, wp_mask], axis = 1))
sst_ep = detrend(np.nanmean(sst_anom[:, ep_mask], axis = 1))
sst_al = detrend(np.nanmean(sst_anom[:, al_mask], axis = 1))

geo_col = np.nanmean(geo_anom_detrend[:, :, np.abs(lat_tropics) <= 20, :], axis = (2, 3))
temp_col = np.nanmean(temp_anom_detrend[:, :, np.abs(lat_tropics) <= 20, :], axis = (2, 3))
geo_col_wp = np.nanmean(geo_anom_detrend[:, :, wp_mask], axis = 2)
temp_col_wp = np.nanmean(temp_anom_detrend[:, :, wp_mask], axis = 2)
geo_col_ep = np.nanmean(geo_anom_detrend[:, :, ep_mask], axis = 2)
temp_col_ep = np.nanmean(temp_anom_detrend[:, :, ep_mask], axis = 2)
geo_col_al = np.nanmean(geo_anom_detrend[:, :, al_mask], axis = 2)
temp_col_al = np.nanmean(temp_anom_detrend[:, :, al_mask], axis = 2)

geo_m = np.zeros((4, len(lvl)))
geo_cor = np.zeros((4, len(lvl)))
temp_m = np.zeros((4, len(lvl)))
temp_cor = np.zeros((4, len(lvl)))
geo_cor_table = np.zeros((4, len(lvl), len(lvl)))
geo_m_table = np.zeros((4, len(lvl), len(lvl)))
for i in range(len(lvl)):
    m, x = np.polyfit(sst_tropics, geo_col[:, i], 1)
    geo_m[0, i] = m
    geo_cor[0, i] = np.corrcoef(sst_tropics, geo_col[:, i])[0, 1]

    m, x = np.polyfit(sst_tropics, temp_col[:, i], 1)
    temp_m[0, i] = m
    temp_cor[0, i] = np.corrcoef(sst_tropics, temp_col[:, i])[0, 1]   

    m, x = np.polyfit(sst_wp, geo_col_wp[:, i], 1)
    geo_m[1, i] = m
    geo_cor[1, i] = np.corrcoef(sst_wp, geo_col_wp[:, i])[0, 1]

    m, x = np.polyfit(sst_wp, temp_col_wp[:, i], 1)
    temp_m[1, i] = m
    temp_cor[1, i] = np.corrcoef(sst_wp, temp_col_wp[:, i])[0, 1]

    m, x = np.polyfit(sst_ep, geo_col_ep[:, i], 1)
    geo_m[2, i] = m
    geo_cor[2, i] = np.corrcoef(sst_ep, geo_col_ep[:, i])[0, 1]

    m, x = np.polyfit(sst_ep, temp_col_ep[:, i], 1)
    temp_m[2, i] = m
    temp_cor[2, i] = np.corrcoef(sst_ep, temp_col_ep[:, i])[0, 1]

    m, x = np.polyfit(sst_al, geo_col_al[:, i], 1)
    geo_m[3, i] = m
    geo_cor[3, i] = np.corrcoef(sst_al, geo_col_al[:, i])[0, 1]

    m, x = np.polyfit(sst_al, temp_col_al[:, i], 1)
    temp_m[3, i] = m
    temp_cor[3, i] = np.corrcoef(sst_al, temp_col_al[:, i])[0, 1]            

    for j in range(len(lvl)):
        geo_m_table[0, i, j] = np.polyfit(geo_col[:, i], geo_col[:, j], 1)[0]
        geo_m_table[1, i, j] = np.polyfit(geo_col_wp[:, i], geo_col_wp[:, j], 1)[0]
        geo_m_table[2, i, j] = np.polyfit(geo_col_ep[:, i], geo_col_ep[:, j], 1)[0]
        geo_m_table[3, i, j] = np.polyfit(geo_col_al[:, i], geo_col_al[:, j], 1)[0]

        geo_cor_table[0, i, j] = np.corrcoef(geo_col[:, i], geo_col[:, j])[0, 1]
        geo_cor_table[1, i, j] = np.corrcoef(geo_col_wp[:, i], geo_col_wp[:, j])[0, 1]  
        geo_cor_table[2, i, j] = np.corrcoef(geo_col_ep[:, i], geo_col_ep[:, j])[0, 1]
        geo_cor_table[3, i, j] = np.corrcoef(geo_col_al[:, i], geo_col_al[:, j])[0, 1]

# %%
geo_lvls = [850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400,
            350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5] 
geo_idxs = [np.argwhere(lvl.data == geo_lvls[i]).flatten()[0] for i in range(len(geo_lvls))]
geo_m_horizontal = np.zeros((len(geo_lvls), 81, 720))
geo_cor_horizontal = np.zeros((len(geo_lvls), 81, 720))
for l in range(len(geo_lvls)):
    print(l)
    for i in range(len(lat_tropics)):
        for j in range(len(lon)):
            m, x = np.polyfit(sst_tropics, geo_anom_detrend[:, geo_idxs[l], i, j], 1)
            geo_m_horizontal[l, i, j] = m
            geo_cor_horizontal[l, i, j] = np.corrcoef(sst_tropics, geo_anom_detrend[:, geo_idxs[l], i, j])[0, 1]  

geo500_m_horizontal = np.zeros((len(geo_lvls), 81, 720))
geo500_cor_horizontal = np.zeros((len(geo_lvls), 81, 720))
for l in range(len(geo_lvls)):
    print(l)
    for i in range(len(lat_tropics)):
        for j in range(len(lon)):
            m, x = np.polyfit(geo_anom_detrend[:, geo_idxs[9], i, j], geo_anom_detrend[:, geo_idxs[l], i, j], 1)
            geo500_m_horizontal[l, i, j] = m
            geo500_cor_horizontal[l, i, j] = np.corrcoef(geo_anom_detrend[:, geo_idxs[9], i, j], geo_anom_detrend[:, geo_idxs[l], i, j])[0, 1]  

# %% Plot geopotential magnitude and correlations with 500-hPa geopotential
plt.rcParams.update({'font.size': 16})
lvl_labels = [1000, 700, 500, 350, 250, 150, 100, 70, 50, 30, 20, 10, 7, 5]

plt.figure(figsize=(8, 6));
plt.plot(geo_m_table[0, 21, :], np.log(lvl))
ax = plt.gca(); plt.grid();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(5), np.log(1000)])
ax.invert_yaxis()
plt.xlabel('Geopotential')
plt.ylabel('Pressure (hPa)')

geo500_m_lat = np.nanmean(geo500_m_horizontal, axis = 2)
geo500_m_lon = np.nanmean(geo500_m_horizontal, axis = 1)

#- np.tile(np.nanmean(geo500_m_lon, axis = 1), (720, 1)).T
plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, np.log(geo_lvls), geo500_m_lon,
               vmin = 0, vmax = 3.5, cmap = 'RdBu_r');
plt.plot(lon, np.zeros(len(lon))+np.log(500), 'k', lw = 2)
plt.colorbar(); ax = plt.gca();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(5), np.log(850)])
ax.invert_yaxis()
plt.xlabel('Longitude')
plt.ylabel('Pressure (hPa)')

plt.figure(figsize=(8, 6));
plt.pcolormesh(lat_tropics, np.log(geo_lvls), geo500_m_lat,
               vmin = 0, vmax = 3, cmap = 'RdBu_r'); 
plt.colorbar(); ax = plt.gca();
plt.plot(lat_tropics, np.zeros(len(lat_tropics))+np.log(500), 'k', lw = 2)
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(5), np.log(850)])
ax.invert_yaxis()
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')

plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, lat_tropics, geo500_m_horizontal[24, :, :],
               vmin = 1, cmap = 'RdBu_r'); 
plt.colorbar();
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, lat_tropics, geo500_m_horizontal[23, :, :],
               vmin = 1, cmap = 'RdBu_r'); 
plt.colorbar();
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, lat_tropics, geo500_m_horizontal[21, :, :],
               vmin = 1, cmap = 'RdBu_r'); 
plt.colorbar();
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, lat_tropics, geo500_m_horizontal[20, :, :],
               vmin = 1, cmap = 'RdBu_r'); 
plt.colorbar();
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, lat_tropics, geo500_m_horizontal[15, :, :],
               vmin = 1, cmap = 'RdBu_r'); 
plt.colorbar();
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# %%
props = dict(boxstyle='round', facecolor='gray', alpha=0.9)
fig, axs = plt.subplots(figsize=(13, 6), ncols = 2, nrows = 2);
cax = axs[0][0].contourf(lon, lat_tropics, geo500_cor_horizontal[20, :, :], 
             levels = np.linspace(0.2, 0.9, 29), cmap = 'plasma', extend = 'both');
axs[0][0].set_ylabel('Latitude $(^\degree)$')
axs[0][0].text(0.0155, 0.865, '100-hPa', transform=axs[0][0].transAxes, fontsize=16,
               verticalalignment='bottom', bbox = props)
axs[0][1].contourf(lon, lat_tropics, geo500_cor_horizontal[21, :, :], 
                   levels = np.linspace(0.2, 0.9, 29), cmap = 'plasma', extend = 'both');
axs[0][1].text(0.0155, 0.865, '70-hPa', transform=axs[0][1].transAxes, fontsize=16,
               verticalalignment='bottom', bbox = props)
axs[1][0].contourf(lon, lat_tropics, geo500_cor_horizontal[22, :, :], 
             levels = np.linspace(0.2, 0.9, 29), cmap = 'plasma', extend = 'both');
axs[1][0].text(0.0155, 0.865, '50-hPa', transform=axs[1][0].transAxes, fontsize=16,
               verticalalignment='bottom', bbox = props)
axs[1][0].set_xlabel('Longitude ($^\degree$E)')
axs[1][0].set_ylabel('Latitude $(^\degree)$')
axs[1][1].contourf(lon, lat_tropics, geo500_cor_horizontal[23, :, :], 
             levels = np.linspace(0.2, 0.9, 29), cmap = 'plasma', extend = 'both');
axs[1][1].text(0.0155, 0.865, '30-hPa', transform=axs[1][1].transAxes, fontsize=16,
               verticalalignment='bottom', bbox = props)
axs[1][1].set_xlabel('Longitude ($^\degree$E)')
cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
fig.colorbar(cax, cax = cbar_ax, ticks = np.linspace(0.2, 0.9, 8))
plt.savefig('/home/jlin/src/strat/geopotential_correlations.png', dpi = 'figure', bbox_inches = 'tight')
# %%

geo_m_lat = np.nanmean(geo_m_horizontal, axis = 2)

plt.figure(figsize=(8, 6));
plt.pcolormesh(lat_tropics, np.log(geo_lvls), geo_m_lat, vmin = 0, vmax = 1050,
            cmap = 'plasma_r'); 
plt.colorbar(); ax = plt.gca();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(5), np.log(850)])
ax.invert_yaxis()
plt.xlabel('Latitude')
plt.ylabel('Pressure (hPa)')


plt.rcParams.update({'font.size': 16})
lvl_labels = [850, 700, 500, 350, 250, 150, 100, 70, 50, 30, 20, 10, 7, 5]

plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, np.log(geo_lvls), geo500_m_lon,
               vmin = 0, vmax = 3.0, cmap = 'plasma_r'); 
plt.colorbar(); ax = plt.gca();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(5), np.log(850)])
ax.invert_yaxis()
plt.xlabel('Longitude')
plt.ylabel('Pressure (hPa)')

geo500_cor_lon = np.nanmean(geo500_cor_horizontal, axis = 1)
plt.figure(figsize=(8, 6));
plt.pcolormesh(lon, np.log(geo_lvls), geo500_cor_lon,
               vmin = 0, vmax = 1, cmap = 'plasma_r'); 
plt.colorbar(); ax = plt.gca();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(5), np.log(850)])
ax.invert_yaxis()
plt.xlabel('Longitude')
plt.ylabel('Pressure (hPa)')


fig, axs = plt.subplots(figsize=(15, 5.5), ncols = 2);
#plt.pcolormesh(lat_tropics, np.log(geo_lvls), geo_m_lat / np.tile(geo_m_lat[:, 40], (81, 1)).T,
#               vmin = 0.75, vmax = 1.0, cmap = 'bone');
ax = axs[0]
cax = ax.contourf(lat_tropics, np.log(geo_lvls), np.minimum(geo_m_lat / np.tile(geo_m_lat[:, 40], (81, 1)).T, 1),
             levels = np.linspace(0.75, 1, 11), cmap = 'bone', extend = 'both'); 
plt.colorbar(cax, ax = ax)
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(10), np.log(850)])
ax.invert_yaxis()
ax.set_xlabel('Latitude')
ax.set_ylabel('Pressure (hPa)')  

#plt.pcolormesh(lat_tropics, np.log(geo_lvls), geo_m_lat / np.tile(geo_m_lat[:, 40], (81, 1)).T,
#               vmin = 0.75, vmax = 1.0, cmap = 'bone');
ax = axs[1]
cax = ax.contourf(lat_tropics, np.log(geo_lvls), np.nanmean(geo_cor_horizontal, axis = 2),
                  levels = np.linspace(0.0, 0.8, 17), cmap = 'gist_heat_r', extend = 'both'); 
plt.colorbar(cax, ax = ax)
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(10), np.log(850)])
ax.invert_yaxis()
plt.xlabel('Latitude')

# %%
plt.figure();
idx_x = 9
xg = geo_m_horizontal[idx_x, :, :].flatten()
yg = geo_m_horizontal[20, :, :].flatten()
round_x = lambda x, i: np.round(x * i) / i
x_bins = np.arange(round_x(np.nanquantile(xg, 0.01), 10), round_x(np.nanquantile(xg, 0.99), 10), 10)
y_bins = np.arange(round_x(np.nanquantile(yg, 0.01), 50), round_x(np.nanquantile(yg, 0.99), 50), 50)
plt.hist2d(xg, yg, bins = (x_bins, y_bins), density = True); 
plt.colorbar()

r_geo_sst = np.zeros(len(geo_lvls))
for i in range(len(geo_lvls)):
    r_geo_sst[i] = np.corrcoef(geo_m_horizontal[i, :, :].flatten(), xg)[0, 1]

lvl_labels = [850, 700, 500, 350, 250, 150, 100, 70, 50, 30, 20, 10, 7, 5]
plt.figure(figsize=(8, 6));
plt.plot(r_geo_sst, np.log(geo_lvls), 'kx-')
ax = plt.gca(); plt.grid();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(10), np.log(850)])
ax.invert_yaxis()
ax.set_xlabel('Correlation')
ax.set_ylabel('Pressure (hPa)')  

plt.figure(figsize=(8, 6));
plt.plot(np.nanmean(geo_m_horizontal, axis = (1, 2)), np.log(geo_lvls), 'kx-')
ax = plt.gca(); plt.grid();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylim([np.log(10), np.log(850)])
ax.invert_yaxis()
ax.set_xlabel('Geopotential')
ax.set_ylabel('Pressure (hPa)')

# %%

# %% Plot geopotential
lvl_labels = [1000, 700, 500, 350, 250, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]
plt.rcParams.update({'font.size': 16})
ls = ['-', '--', '-.', ':']
fig, axs = plt.subplots(figsize=(16, 8), ncols = 3);
for i in range(4):
    axs[0].plot(geo_m[i, :], np.log(lvl), 'b', linestyle=ls[i], linewidth = 2);
    axs[1].plot(temp_m[i, :], np.log(lvl), 'r', linestyle=ls[i], linewidth = 2);
    axs[2].plot(geo_cor[i, :], np.log(lvl), 'b', linestyle=ls[i], linewidth = 2);
    axs[2].plot(temp_cor[i, :], np.log(lvl), 'r', linestyle=ls[i], linewidth = 2);
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='k', lw=2, ls = '-'),
                Line2D([0], [0], color='k', lw=2, ls = '--'),
                Line2D([0], [0], color='k', lw=2, ls = '-.'),
                Line2D([0], [0], color='k', lw=2, ls = ':')]
plt.legend(custom_lines, ['Tropics', 'Indo-Pacific', 'East Pacific', 'Atlantic'],
           ncol = 4, bbox_to_anchor=(0.5, 1.055), fontsize = 16)

for ax in axs:
    ax.grid();
    ax.set_yticks(np.log(lvl_labels))
    ax.set_yticklabels(lvl_labels)
    ax.set_ylim([np.log(10), np.log(1000)])
    ax.invert_yaxis()
axs[0].set_ylabel('Pressure (hPa)')    
axs[0].set_xlim([-250, 1000])
axs[0].set_xlabel('Geopotential ($m^s/s^2$)')
axs[1].set_xlim([-3, 3])
axs[1].set_xlabel('Temperature (K)')
axs[2].set_xlim([-1.0, 1])
axs[2].set_xlabel('Correlation')
plt.savefig('/home/jlin/src/strat/column_regression.png', dpi = 'figure', bbox_inches = 'tight')

fig, axs = plt.subplots(figsize=(16, 5), ncols = 4)
props = dict(boxstyle='round', facecolor='gray', alpha=0.8)
geo_lvls = [750, 500, 100] 
regions = ['Tropics', 'Indo-\nPacific', 'East\nPacific', 'Atlantic']
geo_idxs = [np.argwhere(lvl.data == geo_lvls[i]).flatten()[0] for i in range(3)]
for i in range(4):
    ax = axs[i]
    cols = ['k', 'r', 'b']
    for j in range(len(geo_idxs)):
        ax.plot(geo_cor_table[i, geo_idxs[j], :], np.log(lvl), 'x-', color = cols[j], linewidth = 2);
    ax.grid();
    ax.set_xlabel('Correlation')
    ax.set_xlim([-0.5, 1])
    ax.set_yticks(np.log(lvl_labels))
    ax.set_yticklabels(lvl_labels)
    ax.set_ylim([np.log(10), np.log(1000)])    
    ax.invert_yaxis();
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_xticklabels([-0.5, 0, 0.5, 1])
    ax.yaxis.set_tick_params(labelleft=False)
    ax.text(0.1, 0.1, regions[i], transform=ax.transAxes, fontsize=16,
            verticalalignment='bottom', bbox = props)
axs[0].yaxis.set_tick_params(labelleft=True)    
axs[0].set_ylabel('Pressure (hPa)')
custom_lines = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='b', lw=2)]
plt.legend(custom_lines, ['750', '500', '100'],
           ncol = 4, bbox_to_anchor=(-0.4, 1.2), fontsize = 16)
plt.savefig('/home/jlin/src/strat/geo_correlation.png', dpi = 'figure', bbox_inches = 'tight')

# %%
geo_col_lat = np.nanmean(geo_anom_detrend[:, :, np.abs(lat_tropics) <= 20, :], axis = 3)
lat_bands = np.arange(-20, 20.01, 2)
for i in range(len(lat_bands) - 1):
    mask = (lat_tropics <= lat_bands[i]) & (lat_tropics >= lat_bands[i+1])
    geo_col_band = np.nanmean(geo_col_lat[:, :, mask], axis  = 2)
    sst_band = detrend(np.nanmean(sst_anom[:, mask, :], axis = (1, 2)))



# %% Define warm SST and cold SST tropical events.
sst_tropics = detrend(np.nanmean(sst_anom, axis = (1, 2)))
sst_warm = sst_tropics >= (0.5*np.std(sst_tropics))
sst_cold = sst_tropics <= (-0.5*np.std(sst_tropics))
sst_neutral = np.abs(sst_tropics) <= 1*np.std(sst_tropics)

geo_warm = np.nanmean(geo_anom_detrend[sst_warm, :, :, :], axis = (0, 2, 3))
geo_cold = np.nanmean(geo_anom_detrend[sst_cold, :, :, :], axis = (0, 2, 3))
geo_neutral = np.nanmean(geo_anom_detrend[sst_neutral, :, :, :], axis = (0, 2, 3))

temp_warm = np.nanmean(temp_anom_detrend[sst_warm, :, :, :], axis = (0, 2, 3))
temp_cold = np.nanmean(temp_anom_detrend[sst_cold, :, :, :], axis = (0, 2, 3))
temp_neutral = np.nanmean(temp_anom_detrend[sst_neutral, :, :, :], axis = (0, 2, 3))

#pv_warm = np.nanmean(pv_anom[sst_warm, :, :, :], axis = (0, 2, 3))
#pv_cold = np.nanmean(pv_anom[sst_cold, :, :, :], axis = (0, 2, 3))

# %% Read in ENSO index
import datetime
fn_nino = '/data0/jlin/strat/nino34.long.anom.data'
da_nino = np.loadtxt(fn_nino)
dt_nino = np.array([datetime.datetime(int(x), 1, 1) for x in da_nino[:, 0]])
dt_nino = np.tile(np.expand_dims(dt_nino, 1), (1, 12))
for j in range(dt_nino.shape[0]):
    for i in range(12):
        dt_nino[j, i] = datetime.datetime(dt_nino[j, 0].year, i + 1, 1)
dt_nino = dt_nino.flatten()
nino_idx = da_nino[:, 1:].flatten()

# %% Plot warm and cold events.
plt.figure(figsize=(10, 4)); plt.plot(sst['time'], sst_tropics, 'k');
plt.plot(sst['time'][sst_warm], sst_tropics[sst_warm], 'r.')
plt.plot(sst['time'][sst_cold], sst_tropics[sst_cold], 'b.')

# %% Plot geopotential anomaly profile for warm SST and cold SST months
lvl_labels = [1000, 700, 500, 350, 250, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]
plt.rcParams.update({'font.size': 16})

warm_idxs = np.array([x for x in (np.argwhere(sst_warm).flatten()) if x < sst_warm.shape[0]])
cold_idxs = np.array([x for x in (np.argwhere(sst_cold).flatten()) if x < sst_cold.shape[0]])

anom_total = np.nanmean(sst_tropics[warm_idxs]) - np.nanmean(sst_tropics[cold_idxs])
l_weight = np.nanmean(sst_tropics[warm_idxs]) / anom_total
geo_warm = np.nanmean(geo_anom_detrend[warm_idxs, :, :, :][:, :, np.abs(lat_tropics) <= 15, :], axis = (0, 2, 3))
geo_cold = np.nanmean(geo_anom_detrend[cold_idxs, :, :, :][:, :, np.abs(lat_tropics) <= 15, :], axis = (0, 2, 3))
geo_avg = geo_warm * l_weight - geo_cold * (1 - l_weight)
temp_warm = np.nanmean(temp_anom_detrend[warm_idxs, :, :, :][:, :, np.abs(lat_tropics) <= 15, :], axis = (0, 2, 3))
temp_cold = np.nanmean(temp_anom_detrend[cold_idxs, :, :, :][:, :, np.abs(lat_tropics) <= 15, :], axis = (0, 2, 3))
temp_avg = temp_warm * l_weight - temp_cold * (1 - l_weight)

# correlation coefficients
geo_warm_col = np.nanmean(geo_anom_detrend[warm_idxs, :, :, :][:, :, np.abs(lat_tropics) <= 15, :], axis = (2, 3))
geo_cold_col = np.nanmean(geo_anom_detrend[cold_idxs, :, :, :][:, :, np.abs(lat_tropics) <= 15, :], axis = (2, 3))    
geo_warm_cor = np.zeros((len(lvl), len(lvl)))
geo_cold_cor = np.zeros((len(lvl), len(lvl)))
for i in range(len(lvl)):
    for j in range(len(lvl)):
        geo_warm_cor[i, j] = np.corrcoef(geo_warm_col[:, i], geo_warm_col[:, j])[0, 1]
        geo_cold_cor[i, j] = np.corrcoef(geo_cold_col[:, i], geo_cold_col[:, j])[0, 1]            

geo_warm_trop = np.copy(geo_warm)
geo_cold_trop = np.copy(geo_cold)
temp_warm_trop = np.copy(temp_warm)
temp_cold_trop = np.copy(temp_cold)
geo_warm_cor_trop = np.copy(geo_warm_cor)
geo_cold_cor_trop = np.copy(geo_cold_cor)

# %% Define warm SST and cold SST warm-pool events.
LON, LAT = np.meshgrid(lon, lat[np.abs(lat) <= 20])
wp_mask = (LON >= 75) & (LON <= 180) & (LAT <= 20)

sst_wp = detrend(np.nanmean(sst_anom[:, wp_mask], axis = 1))
sst_wp_warm = sst_wp >= (0.5*np.std(sst_wp))
sst_wp_cold = sst_wp <= (-0.5*np.std(sst_wp))
sst_wp_neutral = np.abs(sst_wp) <= 0.5*np.std(sst_wp)

# geo_wp_warm_profile = np.average(geo_anom[sst_wp_warm, :, :, :][:, :, wp_mask], axis = (0, 2))
# geo_wp_cold_profile = np.nanmean(geo_anom[sst_wp_cold, :, :, :][:, :, wp_mask], axis = (0, 2))
# geo_wp_neutral_profile = np.nanmean(geo_anom[sst_wp_neutral, :, :, :][:, :, wp_mask], axis = (0, 2))

# temp_wp_warm_profile = np.nanmean(temp_anom[sst_wp_warm, :, :, :][:, :, wp_mask], axis = (0, 2))
# temp_wp_cold_profile = np.nanmean(temp_anom[sst_wp_cold, :, :, :][:, :, wp_mask], axis = (0, 2))
# temp_wp_neutral_profile = np.nanmean(temp_anom[sst_wp_neutral, :, :, :][:, :, wp_mask], axis = (0, 2))

# % Plot warm and cold events.
#plt.figure(figsize=(10, 4)); plt.plot(sst['time'], sst_wp, 'k');
#plt.plot(sst['time'][sst_wp_warm], sst_wp[sst_wp_warm], 'r.')
#plt.plot(sst['time'][sst_wp_cold], sst_wp[sst_wp_cold], 'b.')

#warm_idxs = np.array([x for x in (np.argwhere(sst_wp_warm).flatten()) if x < sst_wp_warm.shape[0]])
#cold_idxs = np.array([x for x in (np.argwhere(sst_wp_cold).flatten()) if x < sst_wp_cold.shape[0]])


#anom_total = np.nanmean(sst_wp[warm_idxs]) - np.nanmean(sst_wp[cold_idxs])
#l_weight = np.nanmean(sst_wp[warm_idxs]) / anom_total
#geo_warm_col = np.nanmean(geo_anom_detrend[warm_idxs, :, :, :][:, :, wp_mask], axis = 2)
#geo_cold_col = np.nanmean(geo_anom_detrend[cold_idxs, :, :, :][:, :, wp_mask], axis = 2)

#temp_warm = np.nanmean(temp_anom_detrend[warm_idxs, :, :, :][:, :, wp_mask], axis = (0, 2))
#temp_cold = np.nanmean(temp_anom_detrend[cold_idxs, :, :, :][:, :, wp_mask], axis = (0, 2))
#temp_avg = temp_warm * l_weight - temp_cold * (1 - l_weight)

# %% Plot geopotential
fig, axs = plt.subplots(figsize=(16, 8), ncols = 2);
axs[0].plot(geo_warm_trop, np.log(lvl), 'rx-');
axs[0].plot(geo_warm_wp, np.log(lvl), 'rx--');
axs[0].plot(geo_warm_ep, np.log(lvl), 'rx--');
axs[0].plot(geo_cold_trop, np.log(lvl), 'bx-');
axs[0].plot(geo_cold_wp, np.log(lvl), 'bx-.');
axs[0].plot(geo_cold_ep, np.log(lvl), 'bx-.');

axs[1].plot(temp_warm_trop, np.log(lvl), 'rx-');
axs[1].plot(temp_warm_wp, np.log(lvl), 'rx--');
axs[1].plot(temp_warm_ep, np.log(lvl), 'rx-.');
axs[1].plot(temp_cold_trop, np.log(lvl), 'bx-');
axs[1].plot(temp_cold_wp, np.log(lvl), 'bx--');
axs[1].plot(temp_warm_ep, np.log(lvl), 'rx-.');

for ax in axs:
    ax.grid();
    ax.set_yticks(np.log(lvl_labels))
    ax.set_yticklabels(lvl_labels)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_ylim([np.log(10), np.log(1000)])
    ax.invert_yaxis()
axs[0].set_xlim([-400, 400])
axs[0].set_xlabel('Geopotential ($m^s/s^2$)')
axs[1].set_xlim([-1, 1])
axs[1].set_xlabel('Temperature (K)')

# %% Define warm SST and cold SST East Pacific events.
ep_mask = (LON <= -80) & (LON >= -180) & (LAT <= 20)
sst_ep = detrend(np.nanmean(sst_anom[:, ep_mask], axis = 1))
sst_ep_warm = sst_ep >= (0.5*np.std(sst_ep))
sst_ep_cold = sst_ep <= (-0.5*np.std(sst_ep))
sst_ep_neutral = np.abs(sst_ep) <= 0.5*np.std(sst_ep)

plt.figure(figsize=(10, 4)); plt.plot(sst['time'], sst_ep, 'k');
plt.plot(sst['time'][sst_ep_warm], sst_ep[sst_ep_warm], 'r.')
plt.plot(sst['time'][sst_ep_cold], sst_ep[sst_ep_cold], 'b.')

warm_idxs = np.array([x for x in (np.argwhere(sst_ep_warm).flatten()) if x < sst_ep_warm.shape[0]])
cold_idxs = np.array([x for x in (np.argwhere(sst_ep_cold).flatten()) if x < sst_ep_cold.shape[0]])

anom_total = np.nanmean(sst_ep[warm_idxs]) - np.nanmean(sst_ep[cold_idxs])
l_weight = np.nanmean(sst_ep[warm_idxs]) / anom_total
geo_warm = np.nanmean(geo_anom_detrend[warm_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
geo_cold = np.nanmean(geo_anom_detrend[cold_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
geo_avg = geo_warm * l_weight - geo_cold * (1 - l_weight)
temp_warm = np.nanmean(temp_anom_detrend[warm_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
temp_cold = np.nanmean(temp_anom_detrend[cold_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
temp_avg = temp_warm * l_weight - temp_cold * (1 - l_weight)

# % correlation coefficient
geo_warm_col = np.nanmean(geo_anom_detrend[warm_idxs, :, :, :][:, :, ep_mask], axis = 2)
geo_cold_col = np.nanmean(geo_anom_detrend[cold_idxs, :, :, :][:, :, ep_mask], axis = 2)
geo_warm_cor = np.zeros((len(lvl), len(lvl)))
geo_cold_cor = np.zeros((len(lvl), len(lvl)))
for i in range(len(lvl)):
    for j in range(len(lvl)):
        geo_warm_cor[i, j] = np.corrcoef(geo_warm_col[:, i], geo_warm_col[:, j])[0, 1]
        geo_cold_cor[i, j] = np.corrcoef(geo_cold_col[:, i], geo_cold_col[:, j])[0, 1] 

geo_warm_ep = np.copy(geo_warm)
geo_cold_ep = np.copy(geo_cold)
temp_warm_ep = np.copy(temp_warm)
temp_cold_ep = np.copy(temp_cold)
geo_warm_cor_ep = np.copy(geo_warm_cor)
geo_cold_cor_ep = np.copy(geo_cold_cor)

# %% Plot geopotential
fig, axs = plt.subplots(figsize=(16, 8), ncols = 2);
axs[0].plot(geo_warm_trop, np.log(lvl), 'rx-');
axs[0].plot(geo_warm_wp, np.log(lvl), 'rx--');
axs[0].plot(geo_warm_ep, np.log(lvl), 'rx--');
axs[0].plot(geo_cold_trop, np.log(lvl), 'bx-');
axs[0].plot(geo_cold_wp, np.log(lvl), 'bx-.');
axs[0].plot(geo_cold_ep, np.log(lvl), 'bx-.');

axs[1].plot(temp_warm_trop, np.log(lvl), 'rx-');
axs[1].plot(temp_warm_wp, np.log(lvl), 'rx--');
axs[1].plot(temp_warm_ep, np.log(lvl), 'rx-.');
axs[1].plot(temp_cold_trop, np.log(lvl), 'bx-');
axs[1].plot(temp_cold_wp, np.log(lvl), 'bx--');
axs[1].plot(temp_warm_ep, np.log(lvl), 'rx-.');

for ax in axs:
    ax.grid();
    ax.set_yticks(np.log(lvl_labels))
    ax.set_yticklabels(lvl_labels)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_ylim([np.log(10), np.log(1000)])
    ax.invert_yaxis()
axs[0].set_xlim([-400, 400])
axs[0].set_xlabel('Geopotential ($m^s/s^2$)')
axs[1].set_xlim([-1, 1])
axs[1].set_xlabel('Temperature (K)')

# % Plot correlation matrix
np.hstack((geo_warm_col[:, -16], geo_cold_col[:, -16])),
np.hstack((geo_warm_col[:, 10], geo_cold_col[:, 10]))
plt.figure(figsize=(10, 6))
plt.plot(geo_warm_cor[:, -16], np.log(lvl), 'rx-'); 
plt.plot(geo_cold_cor[:, -16], np.log(lvl), 'bx-'); 
ax = plt.gca(); ax.grid();
ax.set_yticks(np.log(lvl_labels))
ax.set_yticklabels(lvl_labels)
ax.set_ylabel('Pressure (hPa)')
ax.set_ylim([np.log(10), np.log(1000)])
ax.set_xlabel('Correlation (r)')
ax.set_xlim([0, 1])
ax.invert_yaxis(); 

# %% Define warm SST and cold SST atlantic & indian ocean
ep_mask = (LON <= 75) & (LON >= -80) & (LAT <= 20)
sst_ep = detrend(np.nanmean(sst_anom[:, ep_mask], axis = 1))
sst_ep_warm = sst_ep >= (0.5*np.std(sst_ep))
sst_ep_cold = sst_ep <= (-0.5*np.std(sst_ep))
sst_ep_neutral = np.abs(sst_ep) <= 0.5*np.std(sst_ep)

plt.figure(figsize=(10, 4)); plt.plot(sst['time'], sst_ep, 'k');
plt.plot(sst['time'][sst_ep_warm], sst_ep[sst_ep_warm], 'r.')
plt.plot(sst['time'][sst_ep_cold], sst_ep[sst_ep_cold], 'b.')

fig, axs = plt.subplots(figsize=(16, 8), ncols = 2);
for month_lag in month_lags:
    warm_idxs = np.array([x for x in (np.argwhere(sst_ep_warm).flatten() + month_lag) if x < sst_ep_warm.shape[0]])
    cold_idxs = np.array([x for x in (np.argwhere(sst_ep_cold).flatten() + month_lag) if x < sst_ep_cold.shape[0]])

    anom_total = np.nanmean(sst_ep[warm_idxs]) - np.nanmean(sst_ep[cold_idxs])
    l_weight = np.nanmean(sst_ep[warm_idxs]) / anom_total
    geo_warm = np.nanmean(geo_anom_detrend[warm_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
    geo_cold = np.nanmean(geo_anom_detrend[cold_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
    geo_avg = geo_warm * l_weight - geo_cold * (1 - l_weight)
    temp_warm = np.nanmean(temp_anom_detrend[warm_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
    temp_cold = np.nanmean(temp_anom_detrend[cold_idxs, :, :, :][:, :, ep_mask], axis = (0, 2))
    temp_avg = temp_warm * l_weight - temp_cold * (1 - l_weight)

    #norm_fac = geo_avg[np.argwhere(lvl.data == 100).flatten()[0]]
    axs[0].plot(geo_warm, np.log(lvl), 'r-');
    axs[0].plot(geo_cold, np.log(lvl), 'b-');
    axs[0].plot(geo_avg, np.log(lvl), 'k-')

    axs[1].plot(temp_warm, np.log(lvl), 'r-');
    axs[1].plot(temp_cold, np.log(lvl), 'b-');
    axs[1].plot(temp_avg, np.log(lvl), 'k-')
    #geo_std = np.std(np.nanmean(geo_anom[np.concatenate((warm_idxs, cold_idxs)), :, :, :][:, :, np.abs(lat_tropics) <= 15, :], axis = (2,3)), axis = 0)
    #plt.errorbar(geo_avg, np.log(lvl), xerr = geo_std, ecolor = 'k');

for ax in axs:
    ax.grid();
    ax.set_yticks(np.log(lvl_labels))
    ax.set_yticklabels(lvl_labels)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_ylim([np.log(5), np.log(1000)])
    ax.invert_yaxis()
#axs[0].set_xlim([-150, 150])
axs[0].set_xlabel('Geopotential ($m^s/s^2$)')
axs[1].set_xlim([-1, 1])
axs[1].set_xlabel('Temperature (K)')

# %% Plot geopotential anomaly profile for warm SST and cold SST months
anom_total = np.nanmean(sst_tropics[sst_wp_warm]) - np.nanmean(sst_tropics[sst_wp_cold])
l_weight = np.nanmean(sst_tropics[sst_wp_warm]) / anom_total


plt.figure(figsize=(8, 6));
plt.plot(geo_wp_warm_profile, np.log(lvl), 'rx-');
plt.plot(geo_wp_cold_profile, np.log(lvl), 'bx-');
#plt.plot(geo_wp_neutral_profile, np.log(lvl), 'kx-');
#plt.plot(geo_wp_warm_profile*l_weight - geo_wp_cold_profile*(1 - l_weight), np.log(lvl), 'kx-');

plt.gca().set_yticks(np.log(lvl_labels))
plt.gca().set_yticklabels(lvl_labels)
plt.grid(); plt.xlabel('Geopotential Anomaly ($m^2/s^2$)')
plt.ylim([np.log(30), np.log(1000)])
plt.ylabel('Pressure (hPa)')
plt.gca().invert_yaxis()

plt.figure(figsize=(8, 6));
plt.plot(temp_wp_warm_profile, np.log(lvl), 'rx-');
plt.plot(temp_wp_cold_profile, np.log(lvl), 'bx-');

plt.gca().set_yticks(np.log(lvl_labels))
plt.gca().set_yticklabels(lvl_labels)
plt.grid(); plt.xlabel('Temperature Anomaly (K)')
plt.ylim([np.log(30), np.log(1000)])
plt.ylabel('Pressure (hPa)')
plt.gca().invert_yaxis()

# %%
LON, LAT = np.meshgrid(lon, lat[np.abs(lat) <= 20])
lat_t = 15
wp_mask = (LON >= 90) & (LON <= 160) & (np.abs(LAT) <= lat_t)

# % Compare vertical profiles of w between warm pool and tropics
#plt.figure(); plt.plot(w_wp_profile * 1000, np.flip(lvl_z) / 1000, 'rx-');
#plt.plot(w_tropics_profile * 1000, np.flip(lvl_z) / 1000, 'kx-');
#plt.legend(['Warm Pool', 'Tropics']); plt.grid()
#plt.ylim([3, 30]); plt.xlabel('Vertical Velocity (mm/s)'); plt.ylabel('Height (km)')

# % Compare vertical profiles of temperature between warm pool and tropics
month_idx = 5
#temp_wp_profile = np.mean(temp_monthly[month_idx, :, :, :][:, wp_mask], axis = 1)
temp_wp_profile = np.mean(temp_monthly[:, :, wp_mask], axis = (0, 2))
#temp_tropics_profile = np.mean(temp_monthly[month_idx, :, :, :][:, np.abs(lat_tropics) <= lat_t, :], axis = (1, 2))
temp_tropics_profile = np.mean(temp_monthly[:, :, np.abs(lat_tropics) <= lat_t, :], axis = (0, 2, 3))
temp_wp_profiles_months = temp_wp_profile - temp_tropics_profile
geo_wp_profile = np.mean(geo_monthly[month_idx, :, :][:, wp_mask], axis = 1)
geo_tropics_profile = np.mean(geo_monthly[month_idx, :, :, :][:, np.abs(lat_tropics) <= lat_t, :], axis = (1, 2))
geo_wp_profiles_months = geo_wp_profile - geo_tropics_profile
#plt.plot(geo_wp_profile - geo_tropics_profile, np.log(lvl), 'kx-');

plt.figure(figsize=(8,6));
plt.plot(geo_wp_profiles_months, np.log(lvl), 'kx-');
plt.xlabel('Geopotential ($m^2/s^2$)');
plt.gca().set_yticks(np.log(lvl_labels))
plt.gca().set_yticklabels(lvl_labels)
plt.grid(); plt.ylim([np.log(10), np.log(1000)])
plt.ylabel('Pressure (hPa)')
plt.gca().invert_yaxis()

plt.figure(figsize=(8,6));
plt.plot(temp_wp_profiles_months, np.log(lvl), 'kx-');
plt.xlabel('Temperature (K)');
plt.gca().set_yticks(np.log(lvl_labels))
plt.gca().set_yticklabels(lvl_labels)
plt.grid(); plt.ylim([np.log(10), np.log(1000)])
plt.xlim([-2.5, 2.5])
plt.ylabel('Pressure (hPa)')
plt.gca().invert_yaxis()

# %%

wp_mean_temp = np.zeros((12, 37))
wp_mean_geo = np.zeros((12, 37))
for i in range(12):
    wp_mean_temp[i, :] = np.mean(temp_monthly[i::12, :, wp_mask], axis = (0, 2))
    wp_mean_geo[i, :] = np.mean(geo_monthly[i::12, :, wp_mask], axis = (0, 2))    


plt.figure();
plt.plot(range(1, 13), wp_mean_temp[:, 9], 'kx-'); plt.xticks(range(1, 13));

plt.figure();
plt.plot(range(1, 13), wp_mean_geo[:, 9], 'kx-'); plt.xticks(range(1, 13));
# %%
plt.contourf(range(12), lat[np.abs(lat) <= 20], np.nanmean(w_monthly[:, 8, :, :], axis = 2).T * 1000, np.linspace(-0.4, 0.4, 17), cmap = 'RdBu_r'); plt.colorbar()
plt.pcolormesh(range(12), lat[np.abs(lat) <= 20], np.nanmean(temp_monthly[:, 8, :, :], axis = 2).T, cmap = 'RdBu_r'); plt.colorbar()


plt.figure(figsize=(14, 4)); plt.contourf(lon, range(1, 13), np.nanmean(temp_monthly[:, 10, :, :], axis = 1), cmap = 'RdBu_r'); plt.colorbar()
plt.figure(figsize=(14, 4)); plt.contourf(lon, range(1, 13), np.nanmean(w_monthly[:, 8, :, :], axis = 1) * 1000, np.linspace(-0.5, 0.5, 21), cmap = 'RdBu_r'); plt.colorbar()


geo_zonal_anom = geo_monthly[0, 9, :, :] - np.tile(np.nanmean(geo_monthly[0, 9, :, :], axis = 1), (geo_monthly.shape[-1], 1)).T
plt.figure(figsize=(14, 3)); plt.pcolormesh(lon, lat[np.abs(lat) <= 20], geo_zonal_anom, cmap = 'RdBu_r'); plt.colorbar();
plt.figure(figsize=(14, 3)); plt.pcolormesh(lon, lat[np.abs(lat) <= 20], np.nanmean(w_monthly[:, 8, :, :], axis = 0) * 1000, vmin = -0.5, vmax = 0.5, cmap = 'RdBu_r'); plt.colorbar();

# %%
wp_mask = (LON >= 60) & (LON <= 160) & (np.abs(LAT) <= 15)
geo_seasonal_wp = np.zeros(12)
for m in range(12):
    geo_seasonal_wp[m] = np.nanmean(geo_anom_annual[m::12, 9, :, :].data[:, wp_mask])

plt.plot(range(1, 13), geo_seasonal_wp, 'kx-')
plt.xticks(range(1, 13))    ;


# %%
