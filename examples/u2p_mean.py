#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:   
Version:    v1.0
'''

import numpy as np
import check_data
import setup
import model
import poisson
import plot_tools as pt

# -----------------------------------------------------------------------------
# DATA FORMAT REQUIREMENTS
# -----------------------------------------------------------------------------
#
# All velocity and statistical fields must be 2D NumPy arrays defined on a
# shared structured grid. The following rules apply uniformly across all fields.
#
#
# FIELDS
#
# 1. Description
#   x, y              : 1D grid coordinate arrays
#   mask              : Boolean validity mask
#   u, v              : Instantaneous velocity components
#   u_t, v_t          : Temporal derivatives ∂u/∂t, ∂v/∂t
#   u_mean, v_mean    : Time-averaged velocity components
#   u_var, v_var      : Velocity variances
#   uv_mean           : Reynolds shear stress component
#   rho               : Fluid density (scalar)
#   nu                : Kinematic viscosity (scalar)
#   q_0               : Reference free-stream dynamic pressure. If None, the 
#                       reference value is computed as the average over all
#                       data points where Dirichlet boundary conditions are 
#                       enforced.
# 
# 2. Required fields depend on the model
#   NS    : u, v, u_t, v_t, rho, nu
#   TH    : u, v, u_mean, v_mean, rho, nu
#   RANS  : u_mean, v_mean, u_var, v_var, uv_mean, rho, nu
#   all   : x, y, mask, rho, nu, q_0
#
# GRID & COORDINATE SYSTEM
#
# 1. Arrays follow row-major (C) order with a top-left origin
#
#   A[0, 0] : top-left corner of the domain
#   A[row, col] : A[y-index, x-index]
#
# 2. Coordinate arrays
#   x : 1D, length nx, increases left to right
#   y : 1D, length ny, increases top to bottom
#
#
# PREPROCESSING
#
# 1. Vertical flip (bottom-left origin convention)
#    If the source data uses a bottom-left origin, flip vertically before use
#    field = np.flipud(field)
#
# 2. Invalid value encoding
#    Zeros used as sentinels in raw data must be replaced with NaN using
#    field[field == 0] = np.nan
#
# 3. Validity mask
#    A boolean array of shape (ny, nx) that is True at every node where all
#    fields carry valid (non-NaN) data, and False elsewhere.
#
# DATA DICTIONARY
# 
# All variables must be packed into a single dict with the exact keys below:
#
#   data = {
#       'x'      : ...,   # (nx,)    x-coordinates
#       'y'      : ...,   # (ny,)    y-coordinates
#       'domain' : ...,   # (ny, nx) boolean validity mask
#       'bc_mask': ...,   # (ny, nx) boolean set Dirichlet boundary condition
#       'u'      : ...,   # (ny, nx)
#       'v'      : ...,   # (ny, nx)
#       'u_mean' : ...,   # (ny, nx)
#       'v_mean' : ...,   # (ny, nx)
#       'u_var'  : ...,   # (ny, nx)
#       'v_var'  : ...,   # (ny, nx)
#       'uv_mean': ...,   # (ny, nx)
#       'rho'    : ...,   # scalar — fluid density
#       'nu'     : ...,   # scalar — kinematic viscosity
#       'q_0'    : ...,   # scalar — reference free-stream dynamic pressure 
#   }
# -----------------------------------------------------------------------------

# load data
data = np.load('sample_data/P0_alpha15.npy', allow_pickle=True).item()

# check data format
check_data.report(data, 'RANS')

# grid spacing and node connectivity matrix
grid = setup.discretise(data)

# Dirichelet boundary condition relative to free-stream dynamic pressure
bc = setup.dlet(data, 'RANS')

# compute pressure gradient and source fields
fields = model.RANS(data, grid)

# build and solve system of equations
p_mean = poisson.solve(data, grid, bc, fields, wls=False)

# plot result
pt.plot_lic_pressure(
    data['x'],
    data['y'],
    data['u_mean'],
    data['v_mean'],
    p_mean, 
    20,
    1,
    cmap='RdBu_r'
)