import numpy as np
import matplotlib.pyplot as plt

# fn_txt = 'raob.short'
# with open(fn_txt) as f:
#     lines = f.readlines()
# f.close()
#
# lat = np.array([float([y for y in x.split(' ') if len(y) > 0][2]) for x in lines[3:]])
# lon = np.array([float([y for y in x.split(' ') if len(y) > 0][3]) for x in lines[3:]])
# wmo = np.array([float([y for y in x.split(' ') if len(y) > 0][1]) for x in lines[3:]])
#
# fn_inv = 'raobs_inventory.txt'
# with open(fn_inv) as f:
#     lines = f.readlines()
# f.close()
# wmo_inv = np.unique([float([y for y in x.split(' ') if len(y) > 0][0]) for x in lines[10:]])
#
# mask = np.abs(lat) <= 20
# wmo_mask = np.full(np.sum(mask), False)
# for (i,id) in enumerate(wmo[mask]):
#     if id in wmo_inv:
#         wmo_mask[i] = True

# %% Find all raobs stations in the tropics.
fn_stations = 'igra2-station-list.txt'
with open(fn_stations) as f:
    lines = f.readlines()
f.close()
wmo_ids = np.array([[y for y in x.split(' ') if len(y) > 0][0] for x in lines])
wmo_lat = np.array([float([y for y in x.split(' ') if len(y) > 0][1]) for x in lines])
wmo_lon = np.array([float([y for y in x.split(' ') if len(y) > 0][2]) for x in lines])

mask = np.abs(wmo_lat) <= 20
wmo_ids[mask]

#import os
#os.chdir('/nfs/emanuellab001/jzlin/raobs')
#for wmo_id in wmo_ids:
#    cmd = 'curl -O https://www.ncei.noaa.gov/pub/data/igra/data/data-por/%s-data.txt.zip' % wmo_id
#    os.system(cmd)

with open('/nfs/emanuellab001/jzlin/raobs/MYM00048601-data.txt') as f:
    lines = f.readlines()
f.close()

data = []
entry = []
for l in lines:
    if l[0] == '#':
        year = int(l[13:17])
        month = int(l[18:20])
        day = int(l[21:23])
        hour = int(l[24:26])
        entry.insert(0, np.array([year, month, day, hour]))
        if len(entry) > 0:
            data.append(entry)
        entry = []
    else:
        measurement = np.zeros(4)
        measurement[0] = int(l[0])          # level type 1
        measurement[1] = int(l[1])          # level type 2
        measurement[2] = int(l[9:15])       # pressure level
        measurement[3] = int(l[16:21])      # geopotential
        entry.append(measurement)


mask = np.logical_and(np.array(data[-1])[1:, 2] > -10, np.array(data[-1])[1:, 3] > -10)
p_mn = np.array(data[-1])[1:, :][mask, 2]
z_mn = np.array(data[-1])[1:, :][mask, 3]
z_raobs = np.zeros((len(data), len(p_mn)))
for i in range(len(data)):
    data_i = np.array(data[i])
    mask = np.logical_and(data_i[1:, 2] > -10, data_i[1:, 3] > -10)
    p_i = data_i[1:, 2][mask, 2]
    z_i = data_i[1:, 2][mask, 3]
    if len(p_i) >= 2:
        z_raobs[i, :] = np.interp(p_mn, np.flip(p_i), np.flip(z_i), left = np.nan, right = np.nan)
    else:
        z_raobs[i, :] = np.nan

plt.hist(z_raobs[:, -1] / 1000, bins = np.arange(23.4, 24.25, 0.05))
np.sum(~np.isnan(z_raobs[:, -1]))

plt.plot(np.nanmean(z_raobs, axis = 0), p_mn); plt.plot(z_raobs.T, p_mn)
