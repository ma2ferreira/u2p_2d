#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:
Version:    v1.0

Description:
    Pressure-gradient reconstruction models for incompressible flow fields.
    Implements several models for the pressure Poisson source term based
    on instantaneous, Reynolds-averaged, and Taylor-hypothesis formulations of 
    the Navier-Stokes equations.

Includes:
    NS      Pressure reconstruction using the incompressible Navier–Stokes 
            equations.
    TH      Pressure reconstruction using Taylor's frozen turbulence 
            hypothesis.
    RANS    Pressure reconstruction using Reynolds-averaged Navier–Stokes 
            statistics.
'''

import setup

__all__ = [
    'NS',
    'TH',
    'RANS',
]

def NS(data, grid):
    '''
    Reconstruct pressure-gradient components and the pressure Poisson source
    term from the incompressible Navier–Stokes equations using instantaneous
    velocity fields and their temporal derivatives.

    The pressure-gradient components are evaluated as

        p_x = -ρ [
            -∂u/∂t
            + u ∂u/∂x
            + v ∂u/∂y
            - ν (∂²u/∂x² + ∂²u/∂y²)
        ]

        p_y = -ρ [
            -∂v/∂t
            + u ∂v/∂x
            + v ∂v/∂y
            - ν (∂²v/∂x² + ∂²v/∂y²)
        ]

    The pressure Poisson source term is then constructed as

        f = ∇²p = ∂p_x/∂x + ∂p_y/∂y

    PARAMETERS
    data : dict
        Dictionary containing instantaneous flow quantities:
            u : ndarray (2D)
                Instantaneous velocity component in the x-direction.
            v : ndarray (2D)
                Instantaneous velocity component in the y-direction.
            u_t : ndarray (2D)
                Temporal derivative ∂u/∂t.
            v_t : ndarray (2D)
                Temporal derivative ∂v/∂t.
            rho : float
                Fluid density.
            nu : float
                Kinematic viscosity.
    grid : dict
        Dictionary containing finite-difference discretisation data:
            Dx : scipy.sparse.csr_matrix
                Sparse operator approximating ∂/∂x.
            Dy : scipy.sparse.csr_matrix
                Sparse operator approximating ∂/∂y.

    RETURNS
    fields : dict
        Dictionary containing reconstructed pressure quantities:
            p_x : ndarray (2D)
                Pressure gradient in the x-direction.
            p_y : ndarray (2D)
                Pressure gradient in the y-direction.
            f : ndarray (2D)
                Pressure Poisson source term.

    NOTES
    - Spatial derivatives are evaluated using sparse finite-difference
      operators.
    - Interior nodes use central differences; boundaries use one-sided
      differences.
    - Second-order derivatives are obtained by repeated application of the
      gradient operators.
    - Compatible with masked or irregular computational domains.
    - NaN regions are preserved throughout the computation.
    '''
    
    # unpack data
    u = data['u']
    v = data['v']
    u_t = data['u_t']
    v_t = data['v_t']
    rho = data['rho']
    nu = data['nu']
    Dx = grid['Dx']
    Dy = grid['Dy']

    # first-order derivatives
    u_x, u_y = setup.grad_2D(u, Dx, Dy)
    v_x, v_y = setup.grad_2D(v, Dx, Dy)

    # second-order derivatives
    u_xx, _ = setup.grad_2D(u_x, Dx, Dy)
    _, u_yy = setup.grad_2D(u_y, Dx, Dy)
    v_xx, _ = setup.grad_2D(v_x, Dx, Dy)
    _, v_yy = setup.grad_2D(v_y, Dx, Dy)

    # pressure gradient
    p_x = -rho * (- u_t + (u * u_x + v * u_y) - nu * (u_xx + u_yy))
    p_y = -rho * (- v_t + (u * v_x + v * v_y) - nu * (v_xx + v_yy))

    # Poisson equation
    p_xx, _ = setup.grad_2D(p_x, Dx, Dy)
    _, p_yy = setup.grad_2D(p_y, Dx, Dy)
    f = p_xx + p_yy

    # prepare output
    fields = {
        'p_x': p_x,
        'p_y': p_y,
        'f': f
    }
    
    return fields

def TH(data, grid):
    '''
    Reconstruct pressure-gradient components and the pressure Poisson source
    term using the Navier–Stokes equations with Taylor's frozen turbulence
    hypothesis to approximate local temporal derivatives.

    The temporal derivatives are approximated as

        ∂u/∂t ≈ -(u_mean ∂u/∂x + v_mean ∂u/∂y)
        ∂v/∂t ≈ -(u_mean ∂v/∂x + v_mean ∂v/∂y)

    leading to the pressure-gradient expressions

        p_x = -ρ [
            -(u_mean ∂u/∂x + v_mean ∂u/∂y)
            + u ∂u/∂x
            + v ∂u/∂y
            - ν (∂²u/∂x² + ∂²u/∂y²)
        ]

        p_y = -ρ [
            -(u_mean ∂v/∂x + v_mean ∂v/∂y)
            + u ∂v/∂x
            + v ∂v/∂y
            - ν (∂²v/∂x² + ∂²v/∂y²)
        ]

    The pressure Poisson source term is then constructed as

        f = ∇²p = ∂p_x/∂x + ∂p_y/∂y

    PARAMETERS
    data : dict
        Dictionary containing instantaneous and mean flow quantities:
            u : ndarray (2D)
                Instantaneous velocity component in the x-direction.
            v : ndarray (2D)
                Instantaneous velocity component in the y-direction.
            u_mean : ndarray (2D)
                Mean velocity component in the x-direction.
            v_mean : ndarray (2D)
                Mean velocity component in the y-direction.
            rho : float
                Fluid density.
            nu : float
                Kinematic viscosity.
    grid : dict
        Dictionary containing finite-difference discretisation data:
            Dx : scipy.sparse.csr_matrix
                Sparse operator approximating ∂/∂x.
            Dy : scipy.sparse.csr_matrix
                Sparse operator approximating ∂/∂y.

    RETURNS
    fields : dict
        Dictionary containing reconstructed pressure quantities:
            p_x : ndarray (2D)
                Pressure gradient in the x-direction.
            p_y : ndarray (2D)
                Pressure gradient in the y-direction.
            f : ndarray (2D)
                Pressure Poisson source term.

    NOTES
    - Taylor's hypothesis assumes turbulent structures are convected by the
      mean flow with weak temporal evolution.
    - Spatial derivatives are evaluated using sparse finite-difference
      operators.
    - Interior nodes use central differences; boundaries use one-sided
      differences.
    - Second-order derivatives are obtained by repeated application of the
      gradient operators.
    - Compatible with masked or irregular computational domains.
    - NaN regions are preserved throughout the computation.
    '''
    
    # unpack data
    u = data['u']
    v = data['v']
    u_mean = data['u_mean']
    v_mean = data['v_mean']
    rho = data['rho']
    nu = data['nu']
    Dx = grid['Dx']
    Dy = grid['Dy']
    
    # first-order derivatives
    u_x, u_y = setup.grad_2D(u, Dx, Dy)
    v_x, v_y = setup.grad_2D(v, Dx, Dy)

    # second-order derivatives
    u_xx, _ = setup.grad_2D(u_x, Dx, Dy)
    _, u_yy = setup.grad_2D(u_y, Dx, Dy)
    v_xx, _ = setup.grad_2D(v_x, Dx, Dy)
    _, v_yy = setup.grad_2D(v_y, Dx, Dy)

    # pressure gradient
    p_x = -rho * (
        - (u_mean * u_x + v_mean * u_y) 
        + (u * u_x + v * u_y) 
        - nu * (u_xx + u_yy)
    )
    p_y = -rho * (
        - (u_mean * v_x + v_mean * v_y) 
        + (u * v_x + v * v_y) 
        - nu * (v_xx + v_yy)
    )

    # Poisson equation
    p_xx, _ = setup.grad_2D(p_x, Dx, Dy)
    _, p_yy = setup.grad_2D(p_y, Dx, Dy)
    f = p_xx + p_yy
    
    # prepare output
    fields = {
        'p_x': p_x,
        'p_y': p_y,
        'f': f
    }
    
    return fields

def RANS(data, grid):
    '''
    Reconstruct pressure-gradient components and the pressure Poisson source
    term from Reynolds-averaged velocity statistics using the
    Reynolds-Averaged Navier–Stokes (RANS) equations.

    The pressure-gradient components are evaluated as

        p_x = -ρ [
            u_mean ∂u_mean/∂x
          + v_mean ∂u_mean/∂y
          + ∂u_var/∂x
          + ∂uv_mean/∂y
          - ν (∂²u_mean/∂x² + ∂²u_mean/∂y²)
        ]

        p_y = -ρ [
            u_mean ∂v_mean/∂x
          + v_mean ∂v_mean/∂y
          + ∂v_var/∂y
          + ∂uv_mean/∂x
          - ν (∂²v_mean/∂x² + ∂²v_mean/∂y²)
        ]

    The pressure Poisson source term is then constructed as

        f = ∇²p = ∂p_x/∂x + ∂p_y/∂y

    PARAMETERS
    data : dict
        Dictionary containing Reynolds-averaged flow quantities:
            u_mean : ndarray (2D)
                Mean velocity component in the x-direction.
            v_mean : ndarray (2D)
                Mean velocity component in the y-direction.
            u_var : ndarray (2D)
                Velocity variance term u'^2.
            v_var : ndarray (2D)
                Velocity variance term v'^2.
            uv_mean : ndarray (2D)
                Reynolds shear stress term u'v'.
            rho : float
                Fluid density.
            nu : float
                Kinematic viscosity.
    grid : dict
        Dictionary containing finite-difference discretisation data:
            Dx : scipy.sparse.csr_matrix
                Sparse operator approximating ∂/∂x.
            Dy : scipy.sparse.csr_matrix
                Sparse operator approximating ∂/∂y.

    RETURNS
    fields : dict
        Dictionary containing reconstructed pressure quantities:
            p_x : ndarray (2D)
                Pressure gradient in the x-direction.
            p_y : ndarray (2D)
                Pressure gradient in the y-direction.
            f : ndarray (2D)
                Pressure Poisson source term.

    NOTES
    - Spatial derivatives are evaluated using sparse finite-difference
      operators.
    - Interior nodes use central differences; boundaries use one-sided
      differences.
    - Second-order derivatives are obtained by repeated application of the
      gradient operators.
    - Compatible with masked or irregular computational domains.
    - NaN regions are preserved throughout the computation.
    '''
      
    # unpack data
    u_mean = data['u_mean']
    v_mean = data['v_mean']
    u_var = data['u_var']
    v_var = data['v_var']
    uv_mean = data['uv_mean']
    rho = data['rho']
    nu = data['nu']
    Dx = grid['Dx']
    Dy = grid['Dy']

    # first-order derivatives
    u_mean_x, u_mean_y = setup.grad_2D(u_mean, Dx, Dy)
    v_mean_x, v_mean_y = setup.grad_2D(v_mean, Dx, Dy)
    u_var_x, _ = setup.grad_2D(u_var, Dx, Dy)
    _, v_var_y = setup.grad_2D(v_var, Dx, Dy)
    uv_mean_x, uv_mean_y = setup.grad_2D(uv_mean, Dx, Dy)

    # second-order derivatives
    u_mean_xx, _ = setup.grad_2D(u_mean_x, Dx, Dy)
    _, u_mean_yy = setup.grad_2D(u_mean_y, Dx, Dy)
    v_mean_xx, _ = setup.grad_2D(v_mean_x, Dx, Dy)
    _, v_mean_yy = setup.grad_2D(v_mean_y, Dx, Dy)

    # pressure gradient
    p_x = -rho * (
        + u_mean * u_mean_x
        + v_mean * u_mean_y
        + u_var_x
        + uv_mean_y
        - nu * (u_mean_xx + u_mean_yy)
    )
    p_y = -rho * (
        + u_mean * v_mean_x
        + v_mean * v_mean_y
        + v_var_y
        + uv_mean_x
        - nu * (v_mean_xx + v_mean_yy)
    )

    # Poisson equation
    p_xx, _ = setup.grad_2D(p_x, Dx, Dy)
    _, p_yy = setup.grad_2D(p_y, Dx, Dy)
    f = p_xx + p_yy
    
    # prepare output
    fields = {
        'p_x': p_x,
        'p_y': p_y,
        'f': f
    }

    return fields