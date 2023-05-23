# Solves for the coupled troposphere-stratosphere response to a SST
# perturbation, under QE dynamics.
# Modify the input parameters as you see fit, which will change the 
# non-dimensional controlling parameters of the model.
# You will also need to set the meridional SST function, in sT.
# Author: Jonathan Lin
# %%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import findiff

### BEGIN INPUT PARAMETERS ###
k = 0                         # Horizontal wave-number, DOES NOT WORK FOR k != 0 yet

# Dimensional coefficients
Ly = 1200 * 1000              # meridional length scale (m)
beta = 2.3e-11                # meridional gradient of coriolis force
a = 6378 * 1000               # radius of earth
abl = beta * (Ly ** 2) / a
alpha_rad_dim = 30            # stratospheric radiative relaxation, days
nu_dim = 180                  # wave-drag damping time scale, days
H = 16                        # tropopause height
Hs = 8                        # scale height
T_bl = 300                    # surface temperature (K)

# Non-dimensional coefficients
S = 150                       # dry stratification (see paper)
F = 0.1                       # surface friction (see paper)

# Meridional & vertical numerical grid
p_t = 100                     # tropopause pressure (hPa)
p_s = 1000                    # surface pressure (hPa)
Np = 256                      # number of gridpoints in z (troposphere), plotting only
ny = 150                      # number of gridpoints in y (half-hemisphere)
nz = 300                      # number of gridpoints in z (stratosphere)

# Constants
Rd = 287                      # J / (kg K)
Lv = 2.5e6                    # J / kg
cp_d = 1005                   # J / (kg K)
g = 9.81                      # m / s^2
### END INPUT PARAMETERS ###

# Calculate non-dimensional controlling parameters
alpha_rad = 1 / (86400 * alpha_rad_dim) / abl     # radiative relaxation
nu = 1 / (86400 * nu_dim) / abl                   # wave drag

# Set up numerical grid
shape = (ny, nz)
y, z = np.linspace(-10, -10 / ny, shape[0]), np.linspace(1, 6, shape[1])
dy, dz = y[1]-y[0], z[1]-z[0]
Y, Z = np.meshgrid(y, z, indexing='ij')

# INPUT: Set the meridional function of SST (in this case, saturation entropy).
sT = integrate.cumtrapz(Y * (-np.exp(-((Y-2)/0.5)**2) 
                             -np.exp(-((Y+2)/0.5)**2)),
                             x = y, initial = 0, axis = 0)
plt.figure(); plt.plot(y, sT[:, 0], 'k-', zorder = 1); 
plt.plot(np.flip(-y), np.flip(sT[:, 0]), 'k-', zorder = 1)
plt.xlim([-6, 6]); plt.grid(zorder = 0);
plt.xlabel('y (non-dimensional)'); plt.ylabel('Saturation Entropy (n.d.))')

# % Set up conversion from non-dimensional units back to dimensional
# Thermodynamic helper functions
e_sat = lambda T: 6.1094 * np.exp(17.625 * (T - 273) / (T - 273 + 243.04)) * 100;    # saturation vapor pressure (Pa)
q_sat = lambda T, p: 0.622 * e_sat(T) / (p - e_sat(T));                              # saturation specific humidity
gamma_w = lambda T, p: (1 / p) * (Rd * T + Lv * q_sat(T, p)) / (cp_d + ((Lv**2) * q_sat(T, p) * 0.622) / (Rd * (T**2)));

# Tropospheric vertical coordinates
tr_p = np.flip(np.linspace(p_t, p_s, Np))*100;    # troposphere pressure (Pa)
tr_p_nd = tr_p / (tr_p[0] - tr_p[-1])             # non-dimensional pressure
adj_T_mn = H * 1000 / np.log(tr_p_nd[0] / tr_p_nd[-1]) * g / Rd
tr_z = Rd * adj_T_mn / g * np.log(tr_p_nd[0] / tr_p_nd) / 1000

# Stratospheric vertical coordinates (conversion to dimensional)
st_rho_nd = np.exp(H / Hs * (1 - z))
st_z = tr_z[-1] + Hs * (z - 1)

# Moist adiabatic vertical profile
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

# %%
# Set up derivatives
d_dy = findiff.FinDiff(0, dy, 1)
d_dz = findiff.FinDiff(1, dz, 1)

dsdy = d_dy(sT)
U0 = -dsdy / Y     # initial guess is -uBC

if k > 0:
     L = (findiff.Coefficient(-1j * k * S / (Y * alpha_rad)) * findiff.FinDiff(0, dy, 1, acc = 4) +
          findiff.Coefficient(xi / (Y**2)) * findiff.FinDiff(0, dy, 2, acc = 4) -
          findiff.Coefficient(2 * xi / (Y**3)) * findiff.FinDiff(0, dy, 1, acc = 4) +
          findiff.FinDiff(1, dz, 2, acc = 4) -
          H / Hs * findiff.FinDiff(1, dz, 1, acc = 4))
else:
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

# %% Solve for each variable
uBT = U0[:, 0]                                                        # barotropic zonal wind
uBC = dsdy[:, 0] / y                                                  # baroclinic zonal wind
wTp = -alpha_rad / S * dphidz[:, 0]                                   # tropopause vertical velocity
vBT = integrate.cumtrapz(-wTp / B, x = y, initial = 0, axis = 0)      # barotropic meridional wind
vBC = (F * (uBT + uBC) + nu * uBC) / y                                # baroclinic meridional wind
phiBL_y = phi[:, 0] - (1 - nuTp) * sT[:, 0]                           # boundary layer geopotential
phiBT = integrate.cumtrapz(-y * uBT, x = y, initial = 0, axis = 0)    # barotropic geopotential
uS = -d_dy(phi) / Y                                                   # stratospheric zonal wind
vS = uS / Y * nu                                                      # stratospheric meridional wind
wS = -alpha_rad / S * dphidz                                          # stratospheric vertical velocity

# Flip across the equator, since we assume sT is symmetric across the equator.
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

# % Check to see if solution satisfies the linear equations (numerical error)
cmax = 2e-2

# Zonal momentum equation (stratosphere)
znl_mom_eq = Y * vS - nu * uS
# Meridional momentum equation (stratosphere)
mrd_mom_eq = -d_dy(phi) - Y * uS
# Continuity equation
cont_eq = d_dy(vS) + d_dz(wS) - H / Hs * wS
# Thermodynamic equation
thm_eq = wS * S + alpha_rad * d_dz(phi)

plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(figsize = (12, 8), nrows = 2, ncols = 2); 
pcm = axs[0][0].pcolormesh(y, z, znl_mom_eq.T, cmap = 'RdBu_r');
axs[0][0].set_xlabel('y (non-dim)')
axs[0][0].set_ylabel('z (non-dim)')
fig.colorbar(pcm, ax=axs[0][0])

pcm = axs[0][1].pcolormesh(y, z, mrd_mom_eq.T, cmap = 'RdBu_r');
axs[0][1].set_xlabel('y (non-dim)')
axs[0][1].set_ylabel('z (non-dim)')
fig.colorbar(pcm, ax=axs[0][1])

pcm = axs[1][0].pcolormesh(y, z, cont_eq.T, 
                           vmin = -cmax, vmax = cmax, cmap = 'RdBu_r');
axs[1][0].set_xlabel('y (non-dim)')
axs[1][0].set_ylabel('z (non-dim)')
fig.colorbar(pcm, ax=axs[1][0])

pcm = axs[1][0].pcolormesh(y, z, thm_eq.T, cmap = 'RdBu_r');
axs[1][1].set_xlabel('y (non-dim)')
axs[1][1].set_ylabel('z (non-dim)')
fig.colorbar(pcm, ax=axs[1][1])
plt.tight_layout()

# %% Full wind profile
fig, axs = plt.subplots(figsize = (15, 6), ncols = 2);
axs[0].plot([yy[0], yy[-1]], [H, H], 'k-', alpha = 0.5, linewidth = 0.5)
cmax = np.max(u_Mode)
dc = cmax / 11
levels = np.arange(dc / 2, cmax, dc)
axs[0].contour(yy, atm_z, u_Mode.T, colors = 'k', levels = levels);
axs[0].contour(yy, atm_z, u_Mode.T, colors = 'k', levels = -np.flip(levels));
axs[0].set_ylim([0, 35]); axs[0].set_xlim([-5, 5])
axs[0].set_xlabel('y (non-dimensional)')
axs[0].set_ylabel('Height (km)')
axs[0].set_title('Zonal Wind')

cmax = np.max(v_Mode)
dc = cmax / 11
levels_v = np.arange(dc / 2, cmax, dc)
axs[1].plot([yy[0], yy[-1]], [H, H], 'k-', alpha = 0.5, linewidth = 0.5)
axs[1].contour(yy, atm_z, v_Mode.T, colors = 'k', levels = levels_v);
axs[1].contour(yy, atm_z, v_Mode.T, colors = 'k', levels = -np.flip(levels_v));
axs[1].set_ylim([0, 35]); axs[1].set_xlim([-4, 4])
axs[1].set_xlabel('y (non-dimensional)')
axs[1].set_ylabel('Height (km)')
axs[1].set_title('Meridional Wind')
plt.savefig('/home/jlin/src/strat/zonally_symmetric_wind_response.png', dpi = 'figure', bbox_inches = 'tight')

plt.figure(figsize=(16, 8));
cmax = np.max(w_Mode)
plt.plot([yy[0], yy[-1]], [H, H], 'k-', alpha = 0.5, linewidth = 0.5)
plt.contour(yy, atm_z, u_Mode.T, levels = levels, colors = 'k', alpha = 0.75);
h = plt.pcolormesh(yy, atm_z, w_Mode.T, vmin = -cmax, vmax = cmax, cmap = 'RdBu_r');
q_scale = 7
plt.quiver(yy[::10], atm_z[::20], v_Mode[::10, ::20].T * q_scale, np.zeros(v_Mode[::10, ::20].T.shape), scale = 0.8, 
           width = 0.005, headwidth = 2)
plt.colorbar(h, orientation = 'vertical'); plt.ylim([0, 30]); plt.xlim([-4, 4])
plt.xlabel('y (non-dimensional)'); plt.ylabel('Height (km)')
plt.title('Contours: u, Arrows: v, Shading: w')
plt.savefig('/home/jlin/src/strat/zonally_symmetric_response.png', dpi = 'figure', bbox_inches = 'tight')