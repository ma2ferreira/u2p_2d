# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:49:35 2026

@author: mamf1f19
"""

import numpy as np
from scipy.io import loadmat

data1 = loadmat('C:/Users/mamf1f19/OneDrive - Delft University of Technology/MatLab/planar-pressure-reconstruction-main/sample-data/C10U_MEANS.mat')
data2 = loadmat('C:/Users/mamf1f19/OneDrive - Delft University of Technology/MatLab/planar-pressure-reconstruction-main/sample-data/C10U_STATS.mat')
data3 = loadmat('C:/Users/mamf1f19/OneDrive - Delft University of Technology/MatLab/planar-pressure-reconstruction-main/sample-data/M00001.mat')

u_mean = np.flipud(data1['U_MEAN'])
u_mean[u_mean == 0] = np.nan
v_mean = np.flipud(data1['V_MEAN'])
v_mean[v_mean == 0] = np.nan
u_var = np.flipud(data2['U_VAR'])
u_var[u_var == 0] = np.nan
v_var = np.flipud(data2['V_VAR'])
v_var[v_var == 0] = np.nan
uv_mean = np.flipud(data2['uv_MEAN'])
uv_mean[uv_mean == 0] = np.nan
u = np.flipud(data3['U'])
u[u == 0] = np.nan
v = np.flipud(data3['V'])
v[v == 0] = np.nan
domain = ~np.isnan(u_mean)
bc_mask = np.full(u_mean.shape, False)
bc_mask[1206, 1:803] = True
q_0 = 0.5 * 1.225 * 10.12 ** 2

data = {
    'domain': domain,
    'bc_mask': bc_mask,
    'u': u,
    'v': v,
    'u_mean': u_mean,
    'v_mean': v_mean,
    'u_var': u_var,
    'v_var': v_var,
    'uv_mean': uv_mean,
    'x': data1['X'].squeeze(),
    'y': np.flipud(data1['Y'].squeeze()),
    'rho': 1.225,
    'nu': 1e-5,
    'q_0': q_0
}

np.save('C10U.npy', data)
