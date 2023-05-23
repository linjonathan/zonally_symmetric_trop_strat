# %% 
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import findiff

# Horizontal wave-number
k = 0

# Dimensional coefficients
Ly = 1200 * 1000              # meridional length scale (m)
beta = 2.3e-11                # meridional gradient of coriolis force
a = 6378 * 1000               # radius of earth
abl = beta * (Ly ** 2) / a
alpha_rad_dim = 30            # days
nu_dim = 180                  # days

# Non-dimensional coefficients
alpha_rad = 1 / (86400 * alpha_rad_dim) / abl     # radiative relaxation
S = 150                                           # dry stratification
H = 16                                            # tropopause height
Hs = 8                                            # scale height
nu = 1 / (86400 * nu_dim) / abl                   # wave drag
F = 0.1                                           # surface friction

# Meridional & vertical numerical grid
ny = 150
nz = 400
shape = (ny, nz)
y, z = np.linspace(-10, -10 / ny, shape[0]), np.linspace(1, 6, shape[1])
dy, dz = y[1]-y[0], z[1]-z[0]
Y, Z = np.meshgrid(y, z, indexing='ij')

# % Set up conversion from non-dimensional units back to dimensional
# Constants
Rd = 287            # J / (kg K)
Lv = 2.5e6          # J / kg
cp_d = 1005         # J / (kg K)
g = 9.81            # m / s^2

# Thermodynamic helper functions
e_sat = lambda T: 6.1094 * np.exp(17.625 * (T - 273) / (T - 273 + 243.04)) * 100;    # saturation vapor pressure (Pa)
q_sat = lambda T, p: 0.622 * e_sat(T) / (p - e_sat(T));                              # saturation specific humidity
gamma_w = lambda T, p: (1 / p) * (Rd * T + Lv * q_sat(T, p)) / (cp_d + ((Lv**2) * q_sat(T, p) * 0.622) / (Rd * (T**2)));

# Tropospheric vertical coordinates
p_t = 100      # tropopause pressure (hPa)
p_s = 1000     # surface pressure (hPa)
Np = 256       # number of vertical points
tr_p = np.flip(np.linspace(p_t, p_s, Np))*100;    # troposphere pressure (Pa)
tr_p_nd = tr_p / (tr_p[0] - tr_p[-1])             # non-dimensional pressure
adj_T_mn = H * 1000 / np.log(tr_p_nd[0] / tr_p_nd[-1]) * g / Rd
tr_z = Rd * adj_T_mn / g * np.log(tr_p_nd[0] / tr_p_nd) / 1000

# Stratospheric vertical coordinates (conversion to dimensional)
st_rho_nd = np.exp(H / Hs * (1 - z))
st_z = tr_z[-1] + Hs * (z - 1)

# Moist adiabatic vertical profile
T_bl = 300     # surface temperature (K)
T_m = np.zeros(len(tr_p))                         # temperature (K) in troposphere along a moist adiabat
T_m[0] = T_bl
for i in range(len(tr_p)-1):
     T_m[i+1] = T_m[i] + gamma_w(T_m[i], tr_p[i]) * (tr_p[i+1] - tr_p[i])

# Define the barotropic mode (constant)
barotropic_mode = lambda fBT: np.meshgrid(tr_p_nd, fBT)[1]

# Define the first baroclinic mode
T_mn = -np.trapz(T_m, x = tr_p_nd)                # vertically averaged temperature (K)
BC_mode = (T_m - T_mn) / (T_bl - T_mn)
nuT_yp, _ = np.meshgrid(BC_mode, y)
nuTp = BC_mode[-1]
baroclinic_mode = lambda fBC: np.meshgrid(tr_p_nd, fBC)[1] * nuT_yp

# Define additional non-dimensional constants
B = Hs / H * (p_s - p_t) / p_t

# NON-DIMENSIONAL CONTROL PARAMETERS
xi = nu * S / alpha_rad
gamma = alpha_rad / (F * S * B)
print('xi: %f, gamma: %f' % (xi, gamma))

# %% Iterative process to couple with the troposphere
sT = integrate.cumtrapz(Y * (-np.exp(-((Y-2)/0.5)**2) 
                             -np.exp(-((Y+2)/0.5)**2)),
                             x = y, initial = 0, axis = 0)

# Set up derivatives
d_dy = findiff.FinDiff(0, dy, 1)
d_dz = findiff.FinDiff(1, dz, 1)
dsdy = d_dy(sT)
U0 = -dsdy / Y     # initial guess is -uBC

# L = (findiff.Coefficient(-1j * k * S / (Y * alpha_rad)) * findiff.FinDiff(0, dy, 1, acc = 4) +
#      findiff.Coefficient(xi / (Y**2)) * findiff.FinDiff(0, dy, 2, acc = 4) -
#      findiff.Coefficient(2 * xi / (Y**3)) * findiff.FinDiff(0, dy, 1, acc = 4) +
#      findiff.FinDiff(1, dz, 2, acc = 4) -
#      H / Hs * findiff.FinDiff(1, dz, 1, acc = 4))
L = (findiff.Coefficient(xi / (Y**2)) * findiff.FinDiff(0, dy, 2, acc = 4) -
     findiff.Coefficient(2 * xi / (Y**3)) * findiff.FinDiff(0, dy, 1, acc = 4) +
     findiff.FinDiff(1, dz, 2, acc = 4) -
     H / Hs * findiff.FinDiff(1, dz, 1, acc = 4))

f = np.zeros(shape)

# %%
# Use an iterative process so that the troposphere and stratosphere are in balance
# It converges fairly quickly (typically less than 5 iterations).
N_iter = 5
for i in range(N_iter):
    phiT = -integrate.cumtrapz(Y * U0, x = y, initial = 0, axis = 0) - nuTp * sT
    phiT[:, 1:] = np.tile(np.expand_dims(phiT[:, 0], 1), (1, nz - 1))

    bc = findiff.BoundaryConditions(shape)
    bc[0,:] = 0                                  # phi = 0 on left edge
    bc[-1,1:-1] = findiff.FinDiff(0, dy, 1), 0   # dphidy = 0 on right edge
    bc[:, 0] = phiT                              # phi = phiT on bottom edge
    bc[1:-1, -1] = findiff.FinDiff(1, dz, 1), 0  # dphi/dz = 0 on top edge

    pde = findiff.PDE(L, f, bc)
    phi = pde.solve()
    dphidz = d_dz(phi)

    V0df = integrate.cumtrapz(gamma*dphidz, x = y, initial = 0, axis = 0)
    V0df[:, 1:] = np.tile(np.expand_dims(V0df[:, 0], 1), (1, nz - 1))
    U0_new = -Y * V0df - dsdy / Y
    U0_new[:, 1:] = np.tile(np.expand_dims(U0_new[:, 0], 1), (1, nz - 1))
    
    print(np.sum(np.abs(U0 - U0_new)))
    U0 = U0_new

plt.figure()
plt.pcolormesh(y, z, phi.T, cmap = 'gist_heat_r')
plt.colorbar()

# %%
uBT = U0[:, 0]
uBC = dsdy[:, 0] / y
wTp = -alpha_rad / S * dphidz[:, 0]
vBT = integrate.cumtrapz(-wTp / B, x = y, initial = 0, axis = 0)
vBC = (F * (uBT + uBC) + nu * uBC) / y
phiBL_y = phi[:, 0] - (1 - nuTp) * sT[:, 0]
phiBT = integrate.cumtrapz(-y * uBT, x = y, initial = 0, axis = 0)
uS = -d_dy(phi) / Y
vS = uS / Y * nu
wS = -alpha_rad / S * dphidz

# Flip across the equator
yy = np.hstack((y, -np.flip(y)))
eq_flip = lambda f_Mode, sgn : np.vstack((f_Mode, np.flip(sgn * f_Mode, axis = 0)))

phiT_yp = barotropic_mode(phiBL_y) + np.meshgrid(tr_p_nd, sT[:, 0])[1] * (1 - np.meshgrid(BC_mode, y)[0])
phi_Mode = eq_flip(np.hstack((phiT_yp, phi[:, 1:])), 1)

uBT_yz = barotropic_mode(uBT)
uBC_yz = baroclinic_mode(uBC)
vBT_yz = barotropic_mode(vBT)
vBC_yz = baroclinic_mode(vBC)
uT_yz = uBT_yz + uBC_yz
vT_yz = vBT_yz + vBC_yz
u_Mode = eq_flip(np.hstack((uT_yz, uS[:, 1:])), 1)
v_Mode = eq_flip(np.hstack((vT_yz, vS[:, 1:])), -1)
atm_z = np.hstack((tr_z, st_z[1:]))

omegaT_yz = integrate.cumtrapz(-d_dy(vT_yz), x = tr_p_nd, initial = 0, axis = 1)
wT_yz = -Hs / H / tr_p_nd * omegaT_yz
w_Mode = eq_flip(np.hstack((wT_yz, wS[:, 1:])), 1)

#np.savez('/home/jlin/src/strat/eigen_xi%d' % xi, phi_Mode = phi_Mode, w_Mode = w_Mode)
#print('Saved %s' % ('eigen_xi%d' % xi))

# % Check to see if solution satisfies the linear equations
cmax = 2e-2

# Zonal momentum equation (stratosphere)
znl_mom_eq = Y * vS - nu * uS
plt.figure(); plt.pcolormesh(y, z, znl_mom_eq.T, cmap = 'RdBu_r');
plt.colorbar()

# Meridional momentum equation (stratosphere)
mrd_mom_eq = -d_dy(phi) - Y * uS
plt.figure(); plt.pcolormesh(y, z, mrd_mom_eq.T, cmap = 'RdBu_r');
plt.colorbar()

# Continuity equation
plt.figure(); plt.pcolormesh(y, z, (d_dy(vS) + d_dz(wS) - H / Hs * wS).T, 
                             vmin = -cmax, vmax = cmax, cmap = 'RdBu_r');
plt.colorbar()

# Thermodynamic equation
thm_eq = wS * S + alpha_rad * d_dz(phi)
plt.figure(); plt.pcolormesh(y, z, thm_eq.T, cmap = 'RdBu_r');
plt.colorbar()

# %% Full zonal wind profile
plt.rcParams.update({'font.size': 18})
plt.figure();
plt.plot([yy[0], yy[-1]], [H, H], 'k-', alpha = 0.5, linewidth = 0.5)
cmax = np.max(u_Mode)
dc = cmax / 11
levels = np.arange(dc, cmax, dc)
plt.contour(yy, atm_z, u_Mode.T, levels = levels, cmap = 'RdBu_r');
plt.ylim([0, 30]); plt.xlim([-4, 4])

plt.figure();
cmax = np.max(v_Mode)
plt.plot([yy[0], yy[-1]], [H, H], 'k-', alpha = 0.5, linewidth = 0.5)
plt.contour(yy, atm_z, v_Mode.T, levels = 21, cmap = 'RdBu_r');
plt.ylim([0, 30]); plt.xlim([-4, 4])

plt.figure(figsize=(16, 8));
cmax = np.max(w_Mode)
plt.plot([yy[0], yy[-1]], [H, H], 'k-', alpha = 0.5, linewidth = 0.5)
plt.contour(yy, atm_z, u_Mode.T, levels = levels, colors = 'k', alpha = 0.75);
h = plt.pcolormesh(yy, atm_z, w_Mode.T, vmin = -cmax, vmax = cmax, cmap = 'RdBu_r');
#w_Mode_scale = np.copy(w_Mode)
#w_Mode_scale[:, Np:] *= 5
#plt.quiver(yy[::10], atm_z[::20], v_Mode[::10, ::20].T * 5, w_Mode_scale[::10, ::20].T * 5, scale = 1.0, 
#           width = 0.005, headwidth = 2)
q_scale = 7
plt.quiver(yy[::10], atm_z[::20], v_Mode[::10, ::20].T * q_scale, np.zeros(v_Mode[::10, ::20].T.shape), scale = 0.8, 
           width = 0.005, headwidth = 2)
plt.colorbar(h, orientation = 'vertical'); plt.ylim([0, 30]); plt.xlim([-4, 4])
plt.xlabel('y (non-dimensional)'); plt.ylabel('Height (km)')
#plt.savefig('/home/jlin/src/strat/zonally_symmetric_response_xi25.png', dpi = 'figure', bbox_inches = 'tight')

# %%
xis = [1, 10, 25, 100, 1000]
w_Modes = [0]*len(xis)
phi_Modes = [0]*len(xis)
hs = [0]*len(xis)
for i in range(len(xis)):
     phi_Modes[i] = np.load('/home/jlin/src/strat/eigen_xi%d.npz' % xis[i])['phi_Mode']
     w_Modes[i] = np.load('/home/jlin/src/strat/eigen_xi%d.npz' % xis[i])['w_Mode']

cols = ['r', 'm', 'k', 'c', 'b']
fig, axs = plt.subplots(figsize=(20, 7), ncols = 3)

# Interpolate the troposphere to regular z coordinates
nz_int = 1000
z_tr_gr = np.linspace(0, H, nz_int)
d_dz_st = findiff.FinDiff(0, z[1] - z[0], 1, acc = 4)
d_dz_tr = findiff.FinDiff(0, (z_tr_gr[1] - z_tr_gr[0]) / H, 1)
def d_dz_interp(p_Mode):
     ddz_p_Mode_yz = np.zeros((len(yy), nz_int + nz - 1))
     for i in range(ddz_p_Mode_yz.shape[0]):
          ddz_p_Mode_yz[i, 0:nz_int] = d_dz_tr(np.interp(z_tr_gr, tr_z, p_Mode[i, 0:Np]))
          ddz_p_Mode_yz[i, nz_int:] = d_dz_st(p_Mode[i, Np:])
     return ddz_p_Mode_yz

z_gr = np.concatenate((z_tr_gr, (z[1:]-1)*Hs + H))
ax = axs[2]
idx_tp = np.argmin(np.abs(z_gr - H)) + 1
idx_y = np.argmin(np.abs(yy - 1.5))
# Normalize linear modes to T_BL of 1 degree
for i in range(len(xis)):
     idx_eq = np.argmax(w_Modes[i][:, idx_y])
     l_fac = 1 / d_dz_interp(phi_Modes[i])[idx_eq, 0]
     ax.plot(d_dz_interp(phi_Modes[i])[idx_eq, idx_tp:] * l_fac, z_gr[idx_tp:], 
             cols[i], linewidth = 2)
     ax.scatter((d_dz_interp(phi_Modes[i])[idx_eq, :])[idx_tp] * l_fac,
                z_gr[idx_tp], c = cols[i])
ax.plot(d_dz_interp(phi_Modes[i])[idx_eq, 0:idx_tp] * l_fac, z_gr[0:idx_tp], 'k--', linewidth = 2)
ax.set_ylim([0, 30]); ax.set_xlim([-8, 2])
ax.set_xticks(np.arange(-8, 2.01, 2))
ax.set_xlabel('T (non-dimensional)')
ax.set_ylabel('Height (km)')
ax.grid();

ax = axs[0]
for i in range(len(xis)):
     hs[i], = ax.plot(phi_Modes[i][np.argmax(w_Modes[i][:, idx_y]), :] * l_fac, atm_z, cols[i], linewidth = 2)
ax.set_ylim([0, 30]); ax.set_xlim([0, 1.6])
ax.set_xlabel('$\phi$ (non-dimensional)')
ax.set_ylabel('Height (km)')
ax.grid();
ax.legend(hs, [r'$\xi = 1$', r'$\xi = 10$', r'$\xi = 25$',
               r'$\xi = 100$', r'$\xi = 1000$'], ncol = len(xis),
          bbox_to_anchor=(0.7, 1.15), loc="upper left")

ax = axs[1]
for i in range(len(xis)):
     ax.plot(w_Modes[i][np.argmax(w_Modes[i][:, idx_y]), :]  * l_fac, atm_z, cols[i], linewidth = 2)
     max_w = np.max(w_Modes[i][np.argmax(w_Modes[i][:, idx_y]), :])
     idx_max = np.argmax(w_Modes[i][np.argmax(w_Modes[i][:, idx_y]), :])
     idx_mag = np.argwhere(w_Modes[i][np.argmax(w_Modes[i][:, idx_y]), :] <= (max_w / 10)).flatten()
     idx_one_mag = idx_mag[np.argwhere(idx_mag >= idx_max).flatten()[0]]
     ax.scatter(w_Modes[i][np.argmax(w_Modes[i][:, idx_y]), idx_one_mag]  * l_fac, atm_z[idx_one_mag], c = cols[i])
ax.set_ylim([0, 30]); ax.set_xlim([-1e-5, 0.003])
ax.set_xlabel('w (non-dimensional)')
ax.set_ylabel('Height (km)')
ax.grid();

plt.savefig('/home/jlin/src/strat/w_phi_T_profiles.png', dpi = 'figure', bbox_inches = 'tight')

# %% Solve only under a lower boundary condition
phi_Modes = np.zeros((3, ny*2, nz))
wS_Modes = np.zeros((3, ny*2, nz))
xis = [0.1, 1, 100]
sT = integrate.cumtrapz(Y * (-np.exp(-((Y-2)/0.5)**2) 
                             -np.exp(-((Y+2)/0.5)**2)),
                             x = y, initial = 0, axis = 0)

for (i, xi) in enumerate(xis):
     phiT = sT
     L = (findiff.Coefficient(xi / (Y**2)) * findiff.FinDiff(0, dy, 2, acc = 4) -
          findiff.Coefficient(2 * xi / (Y**3)) * findiff.FinDiff(0, dy, 1, acc = 4) +
          findiff.FinDiff(1, dz, 2, acc = 4) -
          H / Hs * findiff.FinDiff(1, dz, 1, acc = 4))

     bc = findiff.BoundaryConditions(shape)
     bc[0,:] = 0                                  # phi = 0 on left edge
     bc[-1,1:-1] = findiff.FinDiff(0, dy, 1), 0   # dphidy = 0 on right edge
     bc[:, 0] = phiT                              # phi = phiT on bottom edge
     bc[1:-1, -1] = findiff.FinDiff(1, dz, 1), 0  # dphi/dz = 0 on top edge

     pde = findiff.PDE(L, f, bc)
     phi = pde.solve()
     dphidz = d_dz(phi)
     wS_Modes[i, :, :] = eq_flip(-alpha_rad / S * dphidz, 1)
     phi_Modes[i, :, :] = eq_flip(phi, 1)

# %%
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(figsize=(15, 8), nrows = 2, ncols = 3)
props = dict(boxstyle='round', facecolor='gray', alpha=0.75)
cmaxs_w = [2e-5, 1.5e-4, 2.5e-3]
for i in range(3):
     ax = axs[0][i]
     Cs = ax.contour(yy, (z-1) * Hs + H, phi_Modes[i, :, :].T, 
                    levels = np.linspace(1e-2, 2, 11), colors = 'k');  
     ax.set_xlim([-6, 6]); ax.set_ylim([16, 55])
     ax.set_xticks(np.arange(-6, 6.01, 2))
     ax.set_yticks([16, 25, 35, 45, 55])
     str = r'$\xi = %.1f$'  % xis[i] if xis[i] < 1 else r'$\xi = %d$' % xis[i]
     ax.text(0.03, 0.95, str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox = props)

     ax = axs[1][i]
     levels = np.linspace(-cmaxs_w[i], cmaxs_w[i], 11)
     Cs = ax.contour(yy, (z-1) * Hs + H, wS_Modes[i, :, :].T, levels = levels, colors = 'k');
     Cs.collections[5].set_color('red')
     ax.set_xlim([-6, 6]); ax.set_ylim([16, 55])
     ax.set_xticks(np.arange(-6, 6.01, 2))
     ax.set_yticks([16, 25, 35, 45, 55])
     str = r'$\xi = %.1f$'  % xis[i] if xis[i] < 1 else r'$\xi = %d$' % xis[i]
     ax.text(0.03, 0.95, str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox = props)     
for ax in axs[1]:
     ax.set_xlabel('y (non-dimensional)')
axs[0][0].set_ylabel('Height (km)')
axs[1][0].set_ylabel('Height (km)')
plt.savefig('/home/jlin/src/strat/stratosphere_xi_geopotential.png', dpi = 'figure', bbox_inches = 'tight')
#plt.plot(phi[-1, :], z); plt.plot(d_dz(phi)[-1, :], z); 
# # TEST

# %%
# Non-dimensional coefficients
alpha_rad = 10.0        # radiative relaxation
S = 100                 # dry stratification
H = 16                  # tropopause height
Hs = 8                  # scale height
nu = 10.0               # wave drag
F = 0.1111              # surface friction
xi = nu * S / alpha_rad
k = 1
print(xi)

sT = integrate.cumtrapz(Y * (-np.exp(-((Y-2)/0.5)**2) 
                             -np.exp(-((Y+2)/0.5)**2)),
                             x = y, initial = 0, axis = 0)
phiT = sT
L = (findiff.Coefficient(1j * k * S / (Y * alpha_rad)) * findiff.FinDiff(0, dy, 1, acc = 4) +
     findiff.Coefficient(xi / (Y**2)) * findiff.FinDiff(0, dy, 2, acc = 4) -
     findiff.Coefficient(2 * xi / (Y**3)) * findiff.FinDiff(0, dy, 1, acc = 4) +
     findiff.FinDiff(1, dz, 2, acc = 4) -
     H / Hs * findiff.FinDiff(1, dz, 1, acc = 4))     
f = np.zeros(shape)
bc = findiff.BoundaryConditions(shape)
bc[0,:] = 0                                  # phi = 0 on left edge
bc[-1,1:-1] = findiff.FinDiff(0, dy, 1), 0   # dphidy = 0 on right edge
bc[:, 0] = phiT                              # phi = phiT on bottom edge
bc[1:-1, -1] = findiff.FinDiff(1, dz, 1), 0  # dphi/dy = 0 on top edge

pde = findiff.PDE(L, f, bc)
phi = pde.solve()
dphidz = d_dz(phi)
phi_Modes = eq_flip(phi, 1)

plt.pcolormesh(yy, z, np.real(phi_Modes).T, vmin = -2, vmax = 2, cmap = 'RdBu_r'); plt.colorbar()
plt.figure(); plt.plot(phi_Modes[150, :], z);

# %%
import xarray as xr
import glob
import datetime
s_dt = np.datetime64(str(datetime.datetime(1979, 1, 1)))
e_dt = np.datetime64(str(datetime.datetime(2021, 12, 31)))

fns = sorted(glob.glob('/data0/jlin/era5/**/*sst_monthly*.nc', recursive = True))
ds = xr.open_mfdataset(fns, concat_dim = "time", combine = "nested", data_vars="minimal").sel(time = slice(s_dt, e_dt))

fns_t = sorted(glob.glob('/data0/jlin/era5/**/*_t_monthly*.nc', recursive = True))
ds_t = xr.open_mfdataset(fns_t, concat_dim = "time", combine = "nested", data_vars="minimal").sel(time = slice(s_dt, e_dt))

sst_mn = np.zeros((12, 181, 360))
for i in range(12):
     sst_mn[i, :, :] = np.nanmean(ds['sst'][i::12, :, :], axis = 0)

lon = ds['longitude']
lat = ds['latitude']
LON, LAT = np.meshgrid(lon, lat)
sst = ds['sst'].sel(time = slice(s_dt, e_dt))
tropical_sst = np.nanmean(sst[:, np.abs(lat) <= 15, :], axis = (1, 2))
tropical_cpt = np.nanmean(np.nanmin(ds_t['t'], axis = 1)[:, np.abs(lat) <= 15, :], axis = (1, 2))

fn_geo = '/data0/jlin/strat/era5/strat_geopotential_monthly.nc'
ds_geo = xr.open_dataset(fn_geo)
tropical_geo = np.nanmean(ds_geo['z'][:, 0, 10, np.abs(ds_geo['latitude']) <= 15, :], axis = (1, 2))
tropical_geo[-1] = np.nan
tropical_geo_70 = np.nanmean(ds_geo['z'][:, 0, 9, np.abs(ds_geo['latitude']) <= 15, :], axis = (1, 2))
tropical_geo_70[-1] = np.nan
tropical_geo_50 = np.nanmean(ds_geo['z'][:, 0, 8, np.abs(ds_geo['latitude']) <= 15, :], axis = (1, 2))
tropical_geo_50[-1] = np.nan
# %%
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(20, 5));

dts = np.array(sst['time'].data.astype('datetime64[s]').tolist())
plt.plot(dts, tropical_sst);
plt.ylabel('Tropical SST (K)')
plt.xlabel('Year')
ax = plt.gca().twinx()
ax.plot(dts, tropical_cpt, 'r');
plt.grid(); plt.gca().invert_yaxis()
ax.set_ylabel('$T_{cpt}$ (K)')

seasonal_tropical_sst = np.zeros(12)
seasonal_tropical_cpt = np.zeros(12)
seasonal_tropical_geo = np.zeros(12)
seasonal_tropical_geo_70 = np.zeros(12)
seasonal_tropical_geo_50 = np.zeros(12)
for i in range(12):
     seasonal_tropical_sst[i] = np.nanmean(tropical_sst[i::12])
     seasonal_tropical_cpt[i] = np.nanmean(tropical_cpt[i::12])
     seasonal_tropical_geo[i] = np.nanmean(tropical_geo[i::12])
     seasonal_tropical_geo_70[i] = np.nanmean(tropical_geo_70[i::12])
     seasonal_tropical_geo_50[i] = np.nanmean(tropical_geo_50[i::12])     

plt.figure(figsize=(8, 6))
h1, = plt.plot(range(1, 13), seasonal_tropical_sst)
plt.xlabel('Month'); plt.ylabel('Tropical SST (K)')
plt.xticks(range(1, 13)); plt.xlim([1, 12])
plt.grid()
ax = plt.gca().twinx()
h2, = ax.plot(range(1, 13), seasonal_tropical_cpt, 'r')
ax.invert_yaxis()
ax.set_ylabel('$T_{cpt}$ (K)')
plt.legend([h1, h2], ['SST', '$T_{cpt}$'])

plt.figure(figsize=(8, 6))
h1, = plt.plot(range(1, 13), seasonal_tropical_cpt)
plt.xlabel('Month'); plt.ylabel('Temperature (K)')
plt.xticks(range(1, 13)); plt.xlim([1, 12])
plt.grid()
ax = plt.gca().twinx()
h2, = ax.plot(range(1, 13), seasonal_tropical_geo, 'r')
ax.plot(range(1, 13), seasonal_tropical_geo_70 - np.nanmean(seasonal_tropical_geo_70) + np.nanmean(seasonal_tropical_geo), 'r--')
ax.plot(range(1, 13), seasonal_tropical_geo_50 - np.nanmean(seasonal_tropical_geo_50) + np.nanmean(seasonal_tropical_geo), 'r-.')
ax.set_ylabel('$\phi (m^2 / s^2)$')
plt.legend([h1, h2], ['$T_{cpt}$', '$\phi$'])