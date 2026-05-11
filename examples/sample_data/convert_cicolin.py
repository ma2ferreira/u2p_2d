# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:49:35 2026

@author: mamf1f19
"""

import numpy as np
from scipy.io import loadmat

data = loadmat('C:/Users/mamf1f19/OneDrive - University of Southampton/Projects/u2p_2d/examples/sample_data/P0_alpha15.mat')

u_mean = np.flipud(data['U_mean'])
u_mean[u_mean == 0] = np.nan
v_mean = np.flipud(data['V_mean'])
v_mean[v_mean == 0] = np.nan
u_var = np.flipud(data['U_RMS'])
u_var[u_var == 0] = np.nan
v_var = np.flipud(data['V_RMS'])
v_var[v_var == 0] = np.nan
uv_mean = np.flipud(data['UV_mean'])
uv_mean[uv_mean == 0] = np.nan
x = data['X'][0,:].squeeze()
y = np.flipud(data['Y'][:,0].squeeze())
mask = ~np.isnan(u_mean)
domain = ~np.isnan(u_mean)
bc_mask = np.full(u_mean.shape, False)
bc_mask[8:188, 6] = True
q_0 = 0.5 * 1000 * 0.28 ** 2

data = {
    'domain': domain,
    'bc_mask': bc_mask,
    'u': None,
    'v': None,
    'u_mean': u_mean,
    'v_mean': v_mean,
    'u_var': u_var,
    'v_var': v_var,
    'uv_mean': uv_mean,
    'x': x,
    'y': y,
    'rho': 1000,
    'nu': 1e-6,
    'q_0': None
}

np.save('P0_alpha15.npy', data)
