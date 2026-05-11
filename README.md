# u2p_2d

Open-source Python implementation of a 2D pressure reconstruction method from velocity field data using a Poisson solver, detailed in:

> Ferreira, M. A. and Ganapathisubramani, B. (2020) *PIV-based pressure estimation in the canopy of urban-like roughness.* Experiments in Fluids, 61(3), 70.

This work was financially supported by the Engineering and Physical Sciences Research Council (EPSRC) through grants EP/P009638/1 and EP/P021476/1.

---

## Installation

This guide covers setup using **Anaconda Prompt** or **Git Bash** on Windows.

### 1. Create a project folder and navigate into it

```bash
mkdir "C:\Users\YourName\Projects"
cd "C:\Users\YourName\Projects"
```

### 2. Clone the repository

```bash
git clone https://github.com/your-username/u2p_2d.git
cd u2p_2d
```

### 3. Create and activate a Conda environment (Python ≥ 3.11)

```bash
conda create --name u2p_env python=3.11
conda activate u2p_env
```

**Using Spyder?** Also install compatible Spyder kernels:

```bash
conda install spyder-kernels==3.1.*
```

### 4. Install the package in editable mode

```bash
pip install -e .
```

Editable mode (`-e`) ensures any changes to the source are immediately reflected without reinstalling.

### 5. Verify the installation

```bash
python
>>> import u2p_2d
>>> exit()
```

### 6. Run the example script

From the repository root:

```bash
python examples/example_script.py
```

---

## Overview

The `u2p_2d` package provides a structured workflow to reconstruct pressure fields from 2D velocity data (e.g. PIV) by solving the pressure Poisson equation on structured grids with optional masked (NaN) regions.

### Key features

**Flexible domain handling**

  The computational domain is defined on a structured grid where invalid regions can be masked using `NaN` values. Nodes are automatically classified into interior, edge, and corner types with consistent neighbour connectivity.

**Sparse finite-difference operators**

 Spatial derivatives are computed using pre-assembled sparse matrices: central differences in the interior and one-sided schemes at boundaries and corners.

**Multiple formulations are available**
 
 Three formulations are supported for computing pressure gradients and Poisson source terms: time-resolved Navier–Stokes (NS),  Reynolds-averaged Navier-Stokes (RANS), and Taylor's Hypothesis (TH), the latter using Taylor's frozen turbulence hypothesis to approximate the time derivative.

### Typical workflow

1. Define the computational domain (with optional masked regions)
2. Build node connectivity (`node_ID`)
3. Compute pressure gradients and Poisson source term from velocity data
4. Solve the Poisson equation for pressure
5. Post-process the reconstructed pressure field

---

## Data Format

Velocity and statistical fields must be provided as 2D NumPy arrays defined on a shared structured grid. The following rules apply uniformly across all fields.

### Fields

#### Description

| Key | Description |
|---|---|
| `x`, `y` | 1D grid coordinate arrays |
| `mask` | Boolean validity mask |
| `u`, `v` | Instantaneous velocity components |
| `u_t`, `v_t` | Temporal derivatives ∂u/∂t, ∂v/∂t |
| `u_mean`, `v_mean` | Time-averaged velocity components |
| `u_var`, `v_var` | Velocity variances |
| `uv_mean` | Reynolds shear stress component |
| `rho` | Fluid density (scalar) |
| `nu` | Kinematic viscosity (scalar) |
| `q_0` | Reference free-stream dynamic pressure (scalar) If `None`, the reference value is computed as the average over all data points where Dirichlet boundary conditions are enforced. |

#### Required fields by formulation

| Formulation | Required fields |
|---|---|
| `NS` | `u`, `v`, `u_t`, `v_t`, `rho`, `nu` |
| `TH` | `u`, `v`, `u_mean`, `v_mean`, `rho`, `nu` |
| `RANS` | `u_mean`, `v_mean`, `u_var`, `v_var`, `uv_mean`, `rho`, `nu` |
| All | `x`, `y`, `mask`, `rho`, `nu`, 'q_0' |

### Grid & Coordinate System

Arrays follow row-major (C) order with a top-left origin:

- `A[0, 0]` — top-left corner of the domain
- `A[row, col]` — maps to `A[y-index, x-index]`

Coordinate arrays:

- `x` — 1D, length `nx`, increases left to right
- `y` — 1D, length `ny`, increases top to bottom

### Preprocessing

**Vertical flip (bottom-left origin convention)**
If the source data uses a bottom-left origin, flip vertically before use:

```python
field = np.flipud(field)
```

**Invalid value encoding**
Zeros used as sentinels in raw data must be replaced with `NaN`:

```python
field[field == 0] = np.nan
```

**Validity mask**
A boolean array of shape `(ny, nx)` that is `True` at every node where all fields carry valid (non-`NaN`) data, and `False` elsewhere.

### Data Dictionary

All variables must be packed into a single dict with the exact keys below:

```python
data = {
    'x'       : ...,   # (nx,)    x-coordinates
    'y'       : ...,   # (ny,)    y-coordinates
    'mask'    : ...,   # (ny, nx) boolean validity mask
    'u'       : ...,   # (ny, nx)
    'v'       : ...,   # (ny, nx)
    'u_mean'  : ...,   # (ny, nx)
    'v_mean'  : ...,   # (ny, nx)
    'u_var'   : ...,   # (ny, nx)
    'v_var'   : ...,   # (ny, nx)
    'uv_mean' : ...,   # (ny, nx)
    'rho'     : ...,   # scalar — fluid density
    'nu'      : ...,   # scalar — kinematic viscosity
    'q_0'     : ...,   # scalar — reference free-stream dynamic pressure
}
```

---

## Troubleshooting

| Issue                           | Fix                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError`           | Confirm the `u2p_env` environment is active and you are in the repo root                               |
| Spyder can't find the package   | Set the interpreter in **Tools → Preferences → Python Interpreter** to the `u2p_env` Python executable |
| `cd` fails on paths with spaces | Wrap the path in double quotes: `cd "C:\My Folder\u2p_2d"`                                             |
