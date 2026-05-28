# u2p_2d

Open-source Python implementation of a 2D pressure reconstruction method from velocity field data using a Poisson solver, detailed in:

> Ferreira, M. A. and Ganapathisubramani, B. (2020) *PIV-based pressure estimation in the canopy of urban-like roughness.* Experiments in Fluids, 61(3), 70.

This work was financially supported by the Engineering and Physical Sciences Research Council (EPSRC) through grants EP/P009638/1 and EP/P021476/1.

---

## Installation

The steps below use **Anaconda Prompt** or **Git Bash**. The package runs on Windows, macOS, and Linux.

### 1. Create a project folder and navigate into it

```bash
mkdir "C:\Users\YourName\Projects"
cd "C:\Users\YourName\Projects"
```

### 2. Clone the repository

```bash
git clone https://github.com/mferreira-dev/u2p_2d.git
cd u2p_2d
```

### 3. Create and activate a Conda environment (Python ≥ 3.11)

```bash
conda create --name u2p_env python=3.11
conda activate u2p_env
```

> **Using Spyder?** Install compatible Spyder kernels before proceeding:
> ```bash
> conda install spyder-kernels
> ```
> Then point Spyder to the `u2p_env` interpreter via **Tools → Preferences → Python Interpreter**.

### 4. Install dependencies

```bash
pip install numpy scipy
```

### 5. Install the package in editable mode

```bash
pip install -e .
```

Editable mode (`-e`) means changes to the source are immediately reflected without reinstalling.

### 6. Verify the installation

```bash
python -c "import u2p_2d; print('u2p_2d imported successfully')"
```

### 7. Run the example script

```bash
python examples/example_script.py
```

---

## Overview

`u2p_2d` reconstructs pressure fields from 2D velocity data (e.g. from PIV measurements) by solving the pressure Poisson equation on a structured grid. Invalid or masked regions are handled transparently via `NaN` encoding.

### Formulations

Three formulations are supported for computing the Poisson source term from velocity data:

| Formulation | `model` key | Description |
|---|---|---|
| Navier–Stokes | `'NS'` | Time-resolved; uses instantaneous velocity and its temporal derivative |
| Taylor's Hypothesis | `'TH'` | Approximates the time derivative via Taylor's frozen turbulence hypothesis |
| Reynolds-averaged NS | `'RANS'` | Uses time-averaged statistics and Reynolds stresses |

### Key features

**Flexible domain handling**
The computational domain is defined on a structured grid. Invalid regions are masked using `NaN` values. Nodes are automatically classified into interior, edge, and corner types with consistent neighbour connectivity.

**Sparse finite-difference operators**
Spatial derivatives are computed using pre-assembled sparse matrices: central differences in the interior and one-sided schemes at boundaries and corners.

### Typical workflow

```
1. Preprocess velocity data (flip, NaN-encode, build domain mask)
2. Pack data into the required dictionary format
3. Validate with validate.report(data, model=...)
4. Build node connectivity matrix (node_ID)
5. Compute pressure gradients and Poisson source term
6. Solve the Poisson equation
7. Post-process the reconstructed pressure field
```

---

## Data Format

Velocity and statistical fields must be provided as 2D NumPy arrays defined on a shared structured grid.

### Fields

| Key | Shape | Description |
|---|---|---|
| `x` | `(nx,)` | Grid x-coordinates (left → right) |
| `y` | `(ny,)` | Grid y-coordinates (top → bottom) |
| `domain` | `(ny,nx)` | Boolean validity mask |
| `bc_mask` | `(ny,nx)` | Dirichlet BC mask — see [Dirichlet Boundary Conditions](#dirichlet-boundary-conditions) |
| `u`, `v` | `(ny,nx)` | Instantaneous velocity components |
| `u_t`, `v_t` | `(ny,nx)` | Temporal derivatives ∂u/∂t, ∂v/∂t |
| `u_mean`, `v_mean` | `(ny,nx)` | Time-averaged velocity components |
| `u_var`, `v_var` | `(ny,nx)` | Velocity variances (must be ≥ 0) |
| `uv_mean` | `(ny,nx)` | Reynolds shear stress |
| `rho` | scalar | Fluid density |
| `nu` | scalar | Kinematic viscosity |
| `q_0` | scalar | Reference free-stream dynamic pressure. Required when `bc_mask` is boolean; if `None`, computed as the average pressure over all Dirichlet nodes. Ignored when `bc_mask` is numeric. |

### Required fields by formulation

| Formulation | Required |
|---|---|
| `NS` | `u`, `v`, `u_t`, `v_t` |
| `TH` | `u`, `v`, `u_mean`, `v_mean` |
| `RANS` | `u_mean`, `v_mean`, `u_var`, `v_var`, `uv_mean` |
| All formulations | `x`, `y`, `domain`, `rho`, `nu` |
| Optional | `bc_mask`, `q_0` |

### Grid & Coordinate System

Arrays follow row-major (C) order with a **top-left origin**:

```
A[0, 0]     → top-left corner
A[row, col] → A[y-index, x-index]
```

- `x` — 1D, length `nx`, strictly increasing left to right
- `y` — 1D, length `ny`, strictly increasing top to bottom

### Preprocessing

**Vertical flip** — if the source data uses a bottom-left origin, flip before use:

```python
field = np.flipud(field)
```

**Sentinel zeros** — zeros used as invalid-node markers in raw data must be replaced with `NaN`:

```python
field[field == 0] = np.nan
```

**Validity mask** — a boolean array of shape `(ny, nx)` that is `True` at every node where all fields carry valid (non-`NaN`) data, and `False` elsewhere:

```python
domain = np.isfinite(u) & np.isfinite(v)   # example
```

### Dirichlet Boundary Conditions

`bc_mask` selects nodes where pressure is prescribed in the Poisson system. Two formats are accepted:

#### 1. Boolean mask — pressure computed from velocity

`bc_mask` is a `(ny, nx)` bool array. `True` marks a Dirichlet node where pressure is derived from the velocity at that node using `q_0` (mandatory in this mode):

| Formulation | Expression |
|---|---|
| `NS` / `TH` | $p = q_0 - \tfrac{1}{2}\rho\left(u^2 + v^2\right)$ |
| `RANS` | $p = q_0 - \tfrac{1}{2}\rho\left(\bar{u}^2 + \bar{v}^2 + \sigma_u^2 + \sigma_v^2\right)$ |

where $\sigma_u^2$ and $\sigma_v^2$ correspond to `u_var` and `v_var`.

#### 2. Numeric mask — pressure prescribed directly

`bc_mask` is a `(ny, nx)` float array. `NaN` marks interior (unconstrained) nodes; non-`NaN` values prescribe pressure directly. `q_0` and `model` are ignored in this mode.

Example — prescribe $p = 0$ along the top row:

```python
bc_mask = np.full((ny, nx), np.nan)
bc_mask[0, :] = 0.0
```

> **Note:** All Dirichlet nodes must fall within the valid domain (`domain == True`).

### Data Dictionary

```python
data = {
    'x'       : ...,   # (nx,)    x-coordinates
    'y'       : ...,   # (ny,)    y-coordinates
    'domain'  : ...,   # (ny, nx) boolean validity mask
    'bc_mask' : ...,   # (ny, nx) bool  — Dirichlet nodes (pressure computed)
                       #          float — Dirichlet nodes (pressure prescribed)
    'u'       : ...,   # (ny, nx)
    'v'       : ...,   # (ny, nx)
    'u_t'     : ...,   # (ny, nx)
    'v_t'     : ...,   # (ny, nx)
    'u_mean'  : ...,   # (ny, nx)
    'v_mean'  : ...,   # (ny, nx)
    'u_var'   : ...,   # (ny, nx)
    'v_var'   : ...,   # (ny, nx)
    'uv_mean' : ...,   # (ny, nx)
    'rho'     : ...,   # scalar — fluid density
    'nu'      : ...,   # scalar — kinematic viscosity
    'q_0'     : ...,   # scalar — reference pressure (boolean bc_mask only)
}
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'u2p_2d'` | Confirm the `u2p_env` environment is active and you ran `pip install -e .` from the repo root |
| `ModuleNotFoundError: No module named 'numpy'` | Run `pip install numpy scipy` inside the active environment |
| Spyder can't find the package | Set the interpreter in **Tools → Preferences → Python Interpreter** to the `u2p_env` Python executable, then restart Spyder |
| `cd` fails on paths with spaces | Wrap the path in double quotes: `cd "C:\My Folder\u2p_2d"` |
| Validation reports NaNs inside domain | Ensure the `domain` mask is built after all preprocessing (flip, sentinel replacement) is complete |
| Validation warns about sentinel zeros | Replace raw invalid values with `NaN` before building the data dict: `field[field == 0] = np.nan` |
| Pressure looks inverted vertically | Source data likely uses a bottom-left origin — apply `np.flipud` to all fields before packing the dict |

---