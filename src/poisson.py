#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:
Version:    v1.0

Description:
    Sparse least-squares solver for the pressure Poisson equation on
    structured 2D grids with masked regions and mixed boundary conditions.
    Implements finite-difference matrix assembly, Neumann boundary treatment,
    optional weighted least-squares reconstruction, and sparse linear-system
    solution.

Includes:
    solve       Solve the pressure Poisson equation using sparse least
                squares reconstruction.

Internal:
    _build_A    Assemble the sparse finite-difference system matrix.
    _build_b    Assemble the right-hand-side source vector.
'''

import setup
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


__all__ = ['solve']

def solve(data, grid, bc, fields, wls=False):
    '''
    Solve the pressure Poisson equation by least squares. Assembles the sparse 
    linear system

        A p = b

    from reconstructed pressure-gradient and Poisson-source fields, restricts
    the system to valid (unmasked) nodes, and solves the normal equations

        Aᵀ A p = Aᵀ b

    using a sparse direct solver. If weighted least squares is enabled, the
    equations are weighted using residual-based confidence weights to reduce
    the influence of outlier vectors.

    PARAMETERS
    data : dict
        Dictionary containing flow-field data. Must include:
            u_mean : ndarray (2D)
                Mean x-velocity field.
            v_mean : ndarray (2D)
                Mean y-velocity field.
    grid : dict
        Dictionary containing grid and operator information. Must include:
            node : SimpleNamespace
                Structured node connectivity and masking information.
            dx : float
                Grid spacing in the x-direction.
            dy : float
                Grid spacing in the y-direction.
    bc : ndarray or None
        Dirichlet boundary condition array of shape (2, N_bc), where:
            bc[0, :] contains flattened node indices
            bc[1, :] contains prescribed pressure values
    fields : dict
        Dictionary containing reconstructed pressure fields:
            p_x : ndarray (2D)
                Pressure gradient in the x-direction.
            p_y : ndarray (2D)
                Pressure gradient in the y-direction.
            f : ndarray (2D)
                Pressure Poisson source term (∇²p).

        If None, the system is solved with Neumann-only conditions.
    wls : bool, optional
        If True, perform weighted least squares using residual-based weights

            w = 1 / (1 + r²)

        where r is the normalised residual field computed from the velocity
        data. Default is False.

    RETURNS
    p : ndarray (2D)
        Reconstructed pressure field. Invalid or masked nodes are returned
        as NaN values.

    NOTES
    - The system matrix is assembled using finite-difference operators with
      implicit Neumann boundary treatment.
    - Invalid nodes are excluded before solving and restored afterward.
    - Weighted least squares improves robustness in regions containing noisy
      or unreliable velocity vectors.
    - The resulting linear system is symmetric positive semi-definite and is
      solved through the normal equations using sparse linear algebra.
    '''
    
    # unpack data
    u_mean = data['u_mean']
    v_mean = data['v_mean']
    node = grid['node']
    
    # create system of equations
    A = _build_A(grid, bc)
    b = _build_b(fields, grid, bc)

    # remove masked (invalid) nodes
    idx = np.where(node.valid_flat)[0]
    A = A.tocsr()[idx][:, idx]
    b = b[idx]
    
    if wls is True:
        # weighted least squares
        r = setup.residuals(u_mean, v_mean, node, epsilon=0.1)
        w = 1.0 / (1.0 + np.asarray(r).ravel()**2)

        # remove masked (invalid) nodes
        w = w[idx]

        Aw = A.multiply(w[:, None])
        ATA = A.T @ Aw
        ATb = A.T @ (b * w)

    else:
        # standard least squares (identity weighting)
        ATA = A.T @ A
        ATb = A.T @ b

    # solve system of equations
    p_sol = spsolve(ATA, ATb)

    # rebuild full field
    p_flat = np.full(u_mean.size, np.nan)
    p_flat[idx] = p_sol
    p = p_flat.reshape(u_mean.shape)
    
    if wls is True:
        return p, r
    else:
        return p

def _build_A(grid, bc):
    '''
    Assemble the sparse finite-difference system matrix for the pressure
    Poisson equation using ghost-point elimination to enforce Neumann
    boundary conditions. The resulting matrix represents the discrete operator 
    associated with

        -∇²p = f

    on a structured Cartesian grid using a 5-point finite-difference stencil.
    Boundary nodes are modified to account for missing neighbours through
    implicit ghost-node elimination.

    PARAMETERS
    grid : dict
        Dictionary containing grid and node information. Must include:
            node : SimpleNamespace
                Structured node connectivity defining interior, edge,
                and corner node groups.
    bc : ndarray or None
        Dirichlet boundary condition array of shape (2, N_bc), where:
            bc[0, :] contains flattened node indices
            bc[1, :] contains prescribed pressure values
        If provided, corresponding rows in the system matrix are replaced
        with identity equations.

    RETURNS
    A : scipy.sparse.csr_matrix
        Sparse system matrix of shape (N, N), where N is the total number
        of grid nodes. The matrix is returned in CSR format.

    NOTES
    - Interior nodes use the standard 5-point Laplacian stencil:
          center = -4
          neighbours = +1
    - Edge nodes use modified stencils where the opposite neighbour
      coefficient is doubled.
    - Corner nodes use doubled coefficients in both valid directions.
    - Grid indexing follows flattened row-major (C-style) ordering.
    - Neumann boundary conditions are incorporated implicitly through
      stencil modification rather than explicit ghost-node storage.
    - Dirichlet conditions are enforced strongly by replacing the
      corresponding matrix rows with identity rows.
    '''
    
    # unpack data
    node = grid['node']
    
    # intialise matrices
    I, J, V = [], [], []

    def add(rows, cols, vals):
        rows = np.asarray(rows).ravel()
        cols = np.asarray(cols).ravel()
        vals = np.asarray(vals).ravel()
        I.extend(rows.tolist())
        J.extend(cols.tolist())
        V.extend(vals.tolist())

    # free (interior)
    if node.free.size > 0:
        c = np.asarray(node.free[0]).ravel()
        n = np.asarray(node.free[1:]).reshape(4, -1)
        add(c, c, [-4] * len(c))
        for k in range(4):
            add(c, n[k], [1] * len(c))

    # edges
    # north
    if node.n.size > 0:
        c = np.asarray(node.n[0]).ravel()
        add(c, c, [-4] * len(c))
        add(c, node.n[1], [1] * len(c)) # e
        add(c, node.n[2], [2] * len(c)) # s
        add(c, node.n[3], [1] * len(c)) # w

    # east
    if node.e.size > 0:
        c = np.asarray(node.e[0]).ravel()
        add(c, c, [-4] * len(c))
        add(c, node.e[1], [1] * len(c)) # n
        add(c, node.e[2], [1] * len(c)) # s
        add(c, node.e[3], [2] * len(c)) # w

    # south
    if node.s.size > 0:
        c = np.asarray(node.s[0]).ravel()
        add(c, c, [-4] * len(c))
        add(c, node.s[1], [2] * len(c)) # n
        add(c, node.s[2], [1] * len(c)) # e
        add(c, node.s[3], [1] * len(c)) # w

    # west
    if node.w.size > 0:
        c = np.asarray(node.w[0]).ravel()
        add(c, c, [-4] * len(c))
        add(c, node.w[1], [1] * len(c)) # n
        add(c, node.w[2], [2] * len(c)) # e
        add(c, node.w[3], [1] * len(c)) # s

    # corners
    def add_corner(block):
        if block.size > 0:
            c = np.asarray(block[0]).ravel()
            add(c, c, [-4] * len(c))
            add(c, block[1], [2] * len(c))
            add(c, block[2], [2] * len(c))
    add_corner(node.ne)
    add_corner(node.se)
    add_corner(node.nw)
    add_corner(node.sw)
    
    I = np.asarray(I, dtype=int)
    J = np.asarray(J, dtype=int)
    V = np.asarray(V, dtype=float)
    
    # apply Dirichlet boundary conditions
    if bc is not None:
        idx = bc[0, :].astype(int)
        n = I.max() + 1
        is_bc = np.zeros(n, dtype=bool)
        is_bc[idx] = True
        mask = ~is_bc[I]
        I = np.concatenate([I[mask], idx])
        J = np.concatenate([J[mask], idx])
        V = np.concatenate([V[mask], np.ones(idx.size)])

    # build sparse matrix
    N = int(max(I.max(initial=0), J.max(initial=0)) + 1)
    A = csr_matrix((V, (I, J)), shape=(N, N))
    
    return A

def _build_b(fields, grid, bc):
    '''
    Construct the right-hand-side vector for the discrete pressure Poisson
    system using finite-difference Neumann boundary conditions with
    ghost-point elimination. The vector corresponds to the discretised Poisson
    equation

        ∇²p = f

    together with pressure-gradient boundary conditions

        ∂p/∂x = p_x
        ∂p/∂y = p_y

    Boundary contributions are incorporated implicitly through modified
    finite-difference stencils at edge and corner nodes.

    PARAMETERS
    fields : dict
        Dictionary containing reconstructed pressure fields:
            p_x : ndarray (2D)
                Pressure gradient in the x-direction.
            p_y : ndarray (2D)
                Pressure gradient in the y-direction.
            f : ndarray (2D)
                Pressure Poisson source term (∇²p).
    grid : dict
        Dictionary containing grid and node information. Must include:
            dx : float
                Grid spacing in the x-direction.
            dy : float
                Grid spacing in the y-direction.
            node : SimpleNamespace
                Structured node connectivity and boundary classification.
    bc : ndarray or None
        Dirichlet boundary condition array of shape (2, N_bc), where:
            bc[0, :] contains flattened node indices
            bc[1, :] contains prescribed pressure values

        If None, no Dirichlet conditions are applied.

    RETURNS
    b : ndarray (1D)
        Right-hand-side vector aligned with the flattened grid indexing.

    NOTES
    - Interior nodes use the standard 5-point Poisson discretisation.
    - Edge and corner nodes incorporate Neumann flux contributions through
      ghost-point elimination.
    - Arrays are flattened using row-major (C-style) indexing.
    - Dirichlet conditions overwrite the corresponding entries directly in b.
    - Sign conventions follow outward normal directions:
          north (+y), south (−y), east (+x), west (−x).
    '''
    # unpack data
    f = fields['f']
    p_x = fields['p_x']
    p_y = fields['p_y']
    dx = grid['dx']
    dy = grid['dy']
    node = grid['node']
    
    # initialise vector
    f_flat = f.ravel()
    p_x_flat = p_x.ravel()
    p_y_flat = p_y.ravel()
    b = np.zeros(f_flat.size)
    
    # free (interior)
    b[node.free[0,:]] = f_flat[node.free[0,:]] * dx * dy
    
    # edges
    if node.n.shape[1] > 0:
        idx = node.n[0]
        b[idx] = f_flat[idx]*dx*dy + 2*dy*p_y_flat[idx]
    if node.e.shape[1] > 0:
        idx = node.e[0]
        b[idx] = f_flat[idx]*dx*dy - 2*dx*p_x_flat[idx]
    if node.s.shape[1] > 0:
        idx = node.s[0]
        b[idx] = f_flat[idx]*dx*dy - 2*dy*p_y_flat[idx]
    if node.w.shape[1] > 0:
        idx = node.w[0]
        b[idx] = f_flat[idx]*dx*dy + 2*dx*p_x_flat[idx]
        
    # corners
    if node.ne.shape[1] > 0:
        idx = node.ne[0]
        b[idx] = f_flat[idx]*dx*dy - 2*dx*p_x_flat[idx] + 2*dy*p_y_flat[idx]
    if node.se.shape[1] > 0:
        idx = node.se[0]
        b[idx] = f_flat[idx]*dx*dy - 2*dx*p_x_flat[idx] - 2*dy*p_y_flat[idx]
    if node.nw.shape[1] > 0:
        idx = node.nw[0]
        b[idx] = f_flat[idx]*dx*dy + 2*dx*p_x_flat[idx] + 2*dy*p_y_flat[idx]
    if node.sw.shape[1] > 0:
        idx = node.sw[0]
        b[idx] = f_flat[idx]*dx*dy + 2*dx*p_x_flat[idx] - 2*dy*p_y_flat[idx]
        
    # apply Dirichlet boundary conditions
    if bc is not None:
        b[bc[0,:].astype(int)] = bc[1,:]
    return b
