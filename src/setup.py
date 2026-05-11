#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:
Version:    v1.0

Description:
    Grid discretisation and numerical utility tools for structured 2D flow
    reconstruction problems. Implements sparse finite-difference operators,
    structured node connectivity, masked-domain handling, Dirichlet boundary
    utilities, and robust residual estimation for vector-field outlier
    detection.

Includes:
    discretise     Construct grid geometry and sparse derivative operators.
    grad_2D        Compute spatial gradients using sparse finite differences.
    dlet           Construct Dirichlet pressure boundary conditions.
    residuals      Compute normalised median residuals for vector outlier
                   detection.

Internal:
    _node_ID       Construct structured node connectivity and boundary groups.
    _grad_matrices Assemble sparse finite-difference gradient operators.
'''

import numpy as np
from scipy.sparse import csr_matrix
from types import SimpleNamespace

__all__ = [
    'discretise',
    'grad_2D',
    'dlet',
    'residuals',
]

def discretise(data):
    '''
    Construct grid geometry and sparse finite-difference operators for a
    structured 2D Cartesian domain with optional masked regions. The function 
    computes uniform grid spacing, classifies nodes according to local boundary
    connectivity, and assembles sparse first-order derivative operators for
    spatial differentiation.

    PARAMETERS
    data : dict
        Dictionary containing grid information. Must include:
            x : ndarray (1D)
                Grid coordinates in the x-direction.
            y : ndarray (1D)
                Grid coordinates in the y-direction.
            domain : ndarray (2D)
                Boolean validity mask defining the computational domain.

    RETURNS
    grid : dict
        Dictionary containing discretisation data:
            node : SimpleNamespace
                Structured node connectivity and boundary classification.
            dx : float
                Grid spacing in the x-direction.
            dy : float
                Grid spacing in the y-direction.
            Dx : scipy.sparse.csr_matrix
                Sparse finite-difference operator approximating ∂/∂x.
            Dy : scipy.sparse.csr_matrix
                Sparse finite-difference operator approximating ∂/∂y.

    NOTES
    - Assumes a uniform Cartesian grid.
    - Boundary handling is embedded directly in the node classification.
    - Sparse operators are compatible with masked or irregular domains.
    - Operators act on flattened row-major (C-style) vectors.
    '''
    
    # unpack data
    domain = data['domain']
    x = data['x']
    y = data['y']
    
    # grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    node = _node_ID(data['domain'])
    
    # sparse finite-difference gradient operators
    Dx, Dy = _grad_matrices(node, domain.shape, dx, dy)
        
    # prepare output
    grid = {
        'node': node,
        'dx': dx,
        'dy': dy,
        'Dx': Dx,
        'Dy': Dy
    }
    
    return grid

def grad_2D(A, Dx, Dy):
    '''
    Compute the spatial gradient of a 2D scalar field using pre-assembled
    sparse finite-difference operators. 
    
    The derivatives are evaluated as
    
        ∂A/∂x ≈ Dx · A
        ∂A/∂y ≈ Dy · A
        
    where Dx and Dy contain finite-difference stencils for interior and
    boundary nodes.

    PARAMETERS
    A : ndarray (2D)
        Scalar field defined on a structured grid. May contain NaN values
        representing masked or invalid regions.
    Dx : scipy.sparse matrix
        Sparse finite-difference operator approximating ∂/∂x.
    Dy : scipy.sparse matrix
        Sparse finite-difference operator approximating ∂/∂y.

    RETURNS
    A_dx : ndarray (2D)
        Approximation of ∂A/∂x on the original grid.
    A_dy : ndarray (2D)
        Approximation of ∂A/∂y on the original grid.

    NOTES
    - Computation is performed on flattened row-major (C-style) vectors.
    - NaN values are temporarily replaced by zeros to avoid propagation
      during sparse matrix multiplication.
    - Original NaN locations are restored after differentiation.
    - Valid finite-difference stencils are assumed not to reference masked
      nodes.
    '''

    A = np.asarray(A, dtype=float)
    flat = A.ravel().copy()

    # Zero-fill NaN cells: safe because valid stencils never reference
    # masked cells (guaranteed by nodeID). Avoids NaN propagation.
    nan_mask = np.isnan(flat)
    flat[nan_mask] = 0.0

    A_dx = (Dx @ flat).reshape(A.shape)
    A_dy = (Dy @ flat).reshape(A.shape)

    # Restore NaN at masked positions (those rows were all-zero in Dx/Dy)
    A_dx.ravel()[nan_mask] = np.nan
    A_dy.ravel()[nan_mask] = np.nan

    return A_dx, A_dy

def dlet(data, model):
    '''
    Construct Dirichlet boundary conditions for the pressure Poisson problem.

    Prescribed pressure values are computed from velocity statistics at nodes
    selected by bc_mask and returned alongside their flattened row-major
    indices for direct insertion into the linear system.

    Parameters
    ----------
    data : dict
        Data dictionary. Required keys:
            bc_mask : (ny, nx) boolean Dirichlet node mask
            rho     : scalar fluid density
            u_mean  : (ny, nx) mean streamwise velocity (all models)
            u, v    : (ny, nx) instantaneous velocities (NS, TH)
            u_mean, v_mean, u_var, v_var : (ny, nx) statistics (RANS)
    model : str
        Pressure gradient formulation. One of 'NS', 'TH', 'RANS'.

    NOTES
    - Grid indexing follows row-major (C-style) ordering.
    - Pressure values are prescribed directly in the linear system.
    '''
    
    # unpack data
    bc_mask = data['bc_mask']
    u = data['u']
    v = data['v']
    u_mean = data['u_mean']
    v_mean = data['v_mean']
    u_var = data['u_var']
    v_var = data['v_var']
    rho = data['rho']
    q_0 = data['q_0']
    
    # linear indices where bc_mask is True
    rows, cols = np.where(bc_mask)
    bc_idx = rows * bc_mask.shape[1] + cols

    # reference pressure values
    if data['q_0'] is None:
        q_0 = 0.5 * rho * np.mean(u_mean[bc_mask] ** 2 + v_mean[bc_mask] ** 2)

    if model in ('NS', 'TH'):
        bc_value = q_0 - 0.5 * rho * (
            + (u[bc_mask]**2 + v[bc_mask]**2)
        )
    elif model == 'RANS':
        bc_value =  q_0 - 0.5 * rho * (
            + (u_mean[bc_mask]**2 + v_mean[bc_mask]**2)
            + (u_var[bc_mask]  + v_var[bc_mask])
        )
    '''
    if model in ('NS', 'TH'):
        bc_value = 0.5 * data['rho'] * (
            np.mean(data['u_mean'][bc_mask])**2
            - (data['u'][bc_mask]**2 + data['v'][bc_mask]**2)
        )
    elif model == 'RANS':
        bc_value = 0.5 * data['rho'] * (
            np.mean(data['u_mean'][bc_mask])**2
            - (data['u_mean'][bc_mask]**2 + data['v_mean'][bc_mask]**2)
            - (data['u_var'][bc_mask]  + data['v_var'][bc_mask])
        )
    '''
    return np.vstack([bc_idx, bc_value])

def residuals(u, v, node, epsilon=0.1):
    '''
    Compute the normalised median residual field for vector outlier detection
    based on the universal outlier detection method of Westerweel & Scarano
    (2005). The method compares each velocity vector against the median of its
    local neighbourhood and normalises the deviation using the local median
    residual magnitude.

    PARAMETERS
    u : ndarray (2D)
        x-component of the velocity field.
    v : ndarray (2D)
        y-component of the velocity field.
    node : SimpleNamespace
        Structured node connectivity generated by `_node_ID`.
    epsilon : float, optional
        Small stabilisation parameter preventing division by zero in regions
        with very small residual magnitudes. Default is 0.1.

    RETURNS
    residual : ndarray (2D)
        Normalised median residual field. Larger values indicate vectors
        that deviate strongly from their local neighbourhood.

    NOTES
    - Residuals are evaluated using local neighbour connectivity.
    - Only nodes with at least two valid neighbours are evaluated.
    - Intended for robust weighting and outlier suppression in PIV data.
    - Grid indexing internally follows flattened row-major ordering.
    '''
    uv      = np.stack([u.ravel(), v.ravel()], axis=1)  # (N, 2)
    r = np.zeros(len(uv))

    blocks = [
        node.free,
        node.n, node.e, node.s, node.w,
        node.ne, node.se, node.nw, node.sw,
    ]

    for block in blocks:
        if block.size == 0 or block.ndim < 2:
            continue

        centers = block[0] # (M,)
        neighbor_set = block[1:] # (K, M)

        # neighbour velocities, NaN where index == -1
        valid = neighbor_set >= 0 # (K, M) bool mask
        safe_idx = np.where(valid, neighbor_set, 0) # replace -1 to avoid OOB
        neigh = np.where(valid[..., None], uv[safe_idx], np.nan) # (K, M, 2)

        # median of neighbours
        Um = np.nanmedian(neigh, axis=0) # (M, 2)

        # residuals
        r0 = np.linalg.norm(uv[centers] - Um, axis=1) # (M,)
        ri = np.linalg.norm(neigh - Um, axis=2) # (K, M)
        rm = np.nanmedian(ri, axis=0) # (M,)

        # only score nodes with >= 2 valid neighbours 
        n_valid = valid.sum(axis=0) # (M,)
        r_star = np.where(n_valid >= 2, r0 / (rm + epsilon), 0.0)
        r[centers] = r_star

    return r.reshape(u.shape)

def _node_ID(domain):
    '''
    Construct structured node connectivity for a 2D finite-difference grid
    with optional masked regions. Valid grid nodes are classified into 
    interior, edge, and corner groups according to local neighbour connectivity
    using a 4-neighbour (von Neumann) stencil. Directional neighbour mappings 
    follow clockwise ordering: [north, east, south, west]

    PARAMETERS
    mask : ndarray (2D)
        Boolean validity mask defining the computational domain, where:
            True  → valid node
            False → masked or invalid node

    RETURNS
    node : SimpleNamespace
        Structured node connectivity container containing:
            valid_flat : ndarray (1D)
                Flattened validity mask.
            free : ndarray
                Interior nodes with full neighbour connectivity:
                    [center, north, east, south, west]
            n, e, s, w : ndarray
                Edge-node groups with one missing neighbour.
            ne, se, sw, nw : ndarray
                Corner-node groups with two valid neighbours.
        All arrays contain flattened row-major node indices.

    NOTES
    - Grid indexing follows row-major (C-style) ordering.
    - Boundary classification is determined automatically from the mask.
    - Neighbour ordering is always: north, east, south, west.
    - Designed for irregular and partially masked computational domains.
    '''
    
    # validity mask
    valid = domain
    valid_flat = valid.ravel()
    
    # identify domain nodes
    idx = np.arange(domain.size).reshape(domain.shape)
    cnode = idx.ravel()[valid_flat]
    
    # identify neighbours of domain nodes 
    idx_pad = np.pad(idx, 1, constant_values=-1)
    nnode = np.stack([
        idx_pad[0:-2, 1:-1].ravel(), 
        idx_pad[1:-1, 2:].ravel(),    
        idx_pad[2:, 1:-1].ravel(),
        idx_pad[1:-1, 0:-2].ravel()
    ])
    nnode = nnode[:, valid_flat]
    
    # identify valid neighbours of domain nodes
    valid_pad = np.pad(valid, 1, constant_values=False)
    is_nnode = np.stack([
        valid_pad[0:-2, 1:-1].ravel(),
        valid_pad[1:-1, 2:].ravel(),
        valid_pad[2:, 1:-1].ravel(),
        valid_pad[1:-1, 0:-2].ravel()
    ])[:, valid_flat]
    
    n_valid = is_nnode.sum(0)
    
    # group free (interior) nodes
    free = np.vstack([cnode[n_valid == 4], nnode[:, n_valid == 4]])
    
    # group edge nodes
    is_edge = n_valid == 3
    is_n = is_edge & ~is_nnode[0]
    is_e = is_edge & ~is_nnode[1]
    is_s = is_edge & ~is_nnode[2]
    is_w = is_edge & ~is_nnode[3]
    
    n = np.vstack([cnode[is_n], nnode[1:, is_n]])
    e = np.vstack([cnode[is_e], nnode[[0, 2, 3], :][:, is_e]])
    s = np.vstack([cnode[is_s], nnode[[0, 1, 3], :][:, is_s]])
    w = np.vstack([cnode[is_w], nnode[:3, is_w]])
    
    # group corner nodes
    is_corner = n_valid == 2
    is_ne = is_corner & ~is_nnode[0] & ~is_nnode[1]
    is_se = is_corner & ~is_nnode[2] & ~is_nnode[1]
    is_sw = is_corner & ~is_nnode[2] & ~is_nnode[3]
    is_nw = is_corner & ~is_nnode[0] & ~is_nnode[3]
    
    ne = np.vstack([cnode[is_ne], nnode[[2, 3], :][:, is_ne]])
    se = np.vstack([cnode[is_se], nnode[[0, 3], :][:, is_se]])
    sw = np.vstack([cnode[is_sw], nnode[[0, 1], :][:, is_sw]])
    nw = np.vstack([cnode[is_nw], nnode[[2, 1], :][:, is_nw]])
    
    return SimpleNamespace(
        valid_flat=valid_flat.astype(int),
        free=free.astype(int),
        n=n.astype(int), 
        e=e.astype(int),
        s=s.astype(int), 
        w=w.astype(int),
        ne=ne.astype(int), 
        se=se.astype(int),
        sw=sw.astype(int), 
        nw=nw.astype(int),
    )

def _grad_matrices(node, shape, dx, dy):
    '''
    Construct sparse finite-difference gradient operators for a structured
    2D Cartesian grid with optional masked regions. 
    
    The operators approximate the first-order spatial derivatives ∂/∂x and ∂/∂y
    using second-order central differences in the interior and first-order 
    one-sided differences at edges and corners. Boundary treatment is 
    determined implicitly through the structured node connectivity generated by
    `_node_ID`.

    PARAMETERS
    node : SimpleNamespace
        Structured node connectivity generated by `_node_ID`.
    shape : tuple of int
        Grid shape:
            (ny, nx)
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.

    RETURNS
    Dx : scipy.sparse.csr_matrix
        Sparse finite-difference operator approximating ∂/∂x.
    Dy : scipy.sparse.csr_matrix
        Sparse finite-difference operator approximating ∂/∂y.

    NOTES
    - Operators act on flattened row-major (C-style) vectors.
    - Boundary stencils are embedded directly into the sparse matrices.
    - Compatible with irregular or partially masked domains.
    - Sparse assembly avoids explicit storage of dense differentiation
      matrices.
    '''
    
    N = shape[0] * shape[1]
    rx, cx, vx = [], [], []
    ry, cy, vy = [], [], []
    def add(r, c, v, rows, cols, val):
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        r.extend(rows.tolist())
        c.extend(cols.tolist())
        v.extend([val] * len(rows))
    # free nodes (central diff)
    c, n, e, s, w = node.free
    add(rx, cx, vx, c, e, +1/(2*dx))
    add(rx, cx, vx, c, w, -1/(2*dx))
    add(ry, cy, vy, c, s, +1/(2*dy))
    add(ry, cy, vy, c, n, -1/(2*dy))
    
    # edges
    # north boundary (S - C)/dy
    c, e, s, w = node.n
    add(rx, cx, vx, c, e, +1/(2*dx))
    add(rx, cx, vx, c, w, -1/(2*dx))
    add(ry, cy, vy, c, s, +1/dy)
    add(ry, cy, vy, c, c, -1/dy)
    # east boundary (C - W)/dx
    c, n, s, w = node.e
    add(rx, cx, vx, c, c, +1/dx)
    add(rx, cx, vx, c, w, -1/dx)
    add(ry, cy, vy, c, s, +1/(2*dy))
    add(ry, cy, vy, c, n, -1/(2*dy))
    # south boundary (C - N)/dy
    c, n, e, w = node.s
    add(rx, cx, vx, c, e, +1/(2*dx))
    add(rx, cx, vx, c, w, -1/(2*dx))
    add(ry, cy, vy, c, c, +1/dy)
    add(ry, cy, vy, c, n, -1/dy)
    # west boundary (E - C)/dx
    c, n, e, s = node.w
    add(rx, cx, vx, c, e, +1/dx)
    add(rx, cx, vx, c, c, -1/dx)
    add(ry, cy, vy, c, s, +1/(2*dy))
    add(ry, cy, vy, c, n, -1/(2*dy))
    
    # corners (one-sided both directions)
    # north-east
    c, s, w = node.ne
    add(rx, cx, vx, c, c, +1/dx)
    add(rx, cx, vx, c, w, -1/dx)
    add(ry, cy, vy, c, s, +1/dy)
    add(ry, cy, vy, c, c, -1/dy)
    # south-east
    c, n, w = node.se
    add(rx, cx, vx, c, c, +1/dx)
    add(rx, cx, vx, c, w, -1/dx)
    add(ry, cy, vy, c, c, +1/dy)
    add(ry, cy, vy, c, n, -1/dy)
    # south-west
    c, n, e = node.sw
    add(rx, cx, vx, c, e, +1/dx)
    add(rx, cx, vx, c, c, -1/dx)
    add(ry, cy, vy, c, c, +1/dy)
    add(ry, cy, vy, c, n, -1/dy)
    # north-west
    c, s, e = node.nw
    add(rx, cx, vx, c, e, +1/dx)
    add(rx, cx, vx, c, c, -1/dx)
    add(ry, cy, vy, c, s, +1/dy)
    add(ry, cy, vy, c, c, -1/dy)
    
    Dx = csr_matrix((vx, (rx, cx)), shape=(N, N))
    Dy = csr_matrix((vy, (ry, cy)), shape=(N, N))
    
    return Dx, Dy