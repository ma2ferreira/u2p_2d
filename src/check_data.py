# -*- coding: utf-8 -*-
"""
Structured data validation module.

Author:     Manuel Ferreira
Version:    v1.2
"""

from __future__ import annotations

import numpy as np
from typing import Literal
import sys


# field specifications --------------------------------------------------------

_ALWAYS_REQUIRED = {'x', 'y', 'domain', 'rho', 'nu'}

_MODEL_REQUIRED: dict[str, set[str]] = {
    'NS':   {'u', 'v', 'u_t', 'v_t'},
    'TH':   {'u', 'v', 'u_mean', 'v_mean'},
    'RANS': {'u_mean', 'v_mean', 'u_var', 'v_var', 'uv_mean'},
}

_2D_FIELDS = {
    'domain',
    'bc_mask',
    'u', 'v',
    'u_t', 'v_t',
    'u_mean', 'v_mean',
    'u_var', 'v_var',
    'uv_mean',
}

_SCALAR_KEYS = {'rho', 'nu', 'q_0'}

_ALL_KNOWN_KEYS = {'x', 'y'} | _2D_FIELDS | _SCALAR_KEYS


# public API ------------------------------------------------------------------

def check_data_format(
    data: dict,
    model: Literal['NS', 'TH', 'RANS'] | None = None,
    *,
    strict: bool = False,
    check_nan_consistency: bool = True,
    check_grid_uniformity: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate *data* against the structured-grid data format requirements.

    Parameters
    ----------
    data : dict
        The data dictionary to validate.
    model : {'NS', 'TH', 'RANS'} or None
        When given, checks that all fields required by that model are
        present and valid.
    strict : bool
        If True, unknown keys are treated as errors rather than warnings.
    check_nan_consistency : bool
        If True, checks for NaN holes inside the domain and non-NaN
        leakage outside it.
    check_grid_uniformity : bool
        If True, warns when x or y spacing is not uniform.

    Returns
    -------
    valid : bool
        True if no errors were found.
    messages : list[str]
        Collected validation messages (RESULT first, then errors,
        warnings, info, ok).
    """

    if not isinstance(data, dict):
        return False, [
            f"ERROR  — 'data' must be a dict, got {type(data).__name__}."
        ]

    errors:   list[str] = []
    warnings: list[str] = []
    info:     list[str] = []
    ok:       list[str] = []

    # 1. grid coordinates -----------------------------------------------------

    nx = ny = None

    for key in ('x', 'y'):

        if key not in data:
            errors.append(
                f"ERROR  — Missing required coordinate array '{key}'."
            )
            continue

        arr = data[key]

        if not isinstance(arr, np.ndarray):
            errors.append(
                f"ERROR  — '{key}' must be a NumPy array, "
                f"got {type(arr).__name__}."
            )
            continue

        if arr.ndim != 1:
            errors.append(
                f"ERROR  — '{key}' must be 1-D, got ndim={arr.ndim}."
            )
            continue

        if not np.issubdtype(arr.dtype, np.floating):
            warnings.append(
                f"WARNING — '{key}' dtype is {arr.dtype}; "
                "expected a floating-point type."
            )

        if len(arr) < 2:
            errors.append(
                f"ERROR  — '{key}' must have length ≥ 2, got {len(arr)}."
            )
            continue

        delta = np.diff(arr)

        if not np.all(delta > 0):
            errors.append(
                f"ERROR  — '{key}' must be strictly increasing "
                "(left→right / top→bottom convention)."
            )

        if check_grid_uniformity and delta.size > 1:
            rel_std = np.std(delta) / np.mean(delta)
            if rel_std > 1e-3:
                warnings.append(
                    f"WARNING — '{key}'-grid is not uniform "
                    f"(relative spacing std = {rel_std:.2e})."
                )

        if key == 'x':
            nx = len(arr)
            ok.append(f"OK     — 'x' is a valid 1-D coordinate array, nx={nx}.")
        else:
            ny = len(arr)
            ok.append(f"OK     — 'y' is a valid 1-D coordinate array, ny={ny}.")

    if nx is not None and ny is not None:
        info.append(
            f"INFO   — Grid: nx={nx}, ny={ny} → field shape ({ny}, {nx})."
        )

    # 2. 2-D field helper -----------------------------------------------------

    def check_2d(key: str, *, required: bool) -> bool:
        """Return True if the field passes all 2-D checks."""

        if key not in data:
            if required:
                errors.append(
                    f"ERROR  — Missing required 2-D field '{key}'."
                )
            return False

        arr = data[key]

        if not isinstance(arr, np.ndarray):
            errors.append(
                f"ERROR  — '{key}' must be a NumPy array, "
                f"got {type(arr).__name__}."
            )
            return False

        if arr.ndim != 2:
            errors.append(
                f"ERROR  — '{key}' must be 2-D, got ndim={arr.ndim}."
            )
            return False

        if ny is not None and nx is not None and arr.shape != (ny, nx):
            errors.append(
                f"ERROR  — '{key}' shape {arr.shape} does not match "
                f"(ny={ny}, nx={nx})."
            )
            return False

        ok.append(
            f"OK     — '{key}' shape={arr.shape}, dtype={arr.dtype}."
        )
        return True

    # 3. domain ---------------------------------------------------------------

    domain_ok = check_2d('domain', required=True)

    if domain_ok:
        mask = data['domain']
        if mask.dtype != bool:
            errors.append(
                f"ERROR  — 'domain' must have dtype bool, got {mask.dtype}. "
                "Cast with: data['domain'] = data['domain'].astype(bool)"
            )
            domain_ok = False
        else:
            n_valid = int(mask.sum())
            ok[-1] += f", valid nodes={n_valid}/{mask.size}."

    # 4. bc_mask --------------------------------------------------------------

    bc_mask_mode: str | None = None   # 'boolean' | 'numeric' | None

    if 'bc_mask' not in data:
        warnings.append(
            "WARNING — 'bc_mask' not present; "
            "Dirichlet BC nodes will not be explicitly set."
        )
    else:
        bc_ok = check_2d('bc_mask', required=False)

        if bc_ok:
            arr = data['bc_mask']

            if np.issubdtype(arr.dtype, np.bool_):
                bc_mask_mode = 'boolean'
                n_bc = int(arr.sum())
                ok[-1] += f", mode=boolean, Dirichlet nodes={n_bc}."

                # bc_mask nodes must lie within the valid domain
                if domain_ok:
                    outside = arr & ~data['domain']
                    if outside.any():
                        warnings.append(
                            f"WARNING — 'bc_mask' has {int(outside.sum())} "
                            "Dirichlet node(s) outside the valid domain."
                        )

            elif np.issubdtype(arr.dtype, np.floating):
                bc_mask_mode = 'numeric'
                n_bc = int(np.sum(~np.isnan(arr)))
                ok[-1] += f", mode=numeric, Dirichlet nodes={n_bc}."

                # bc_mask nodes must lie within the valid domain
                if domain_ok:
                    outside = ~np.isnan(arr) & ~data['domain']
                    if outside.any():
                        warnings.append(
                            f"WARNING — 'bc_mask' has {int(outside.sum())} "
                            "prescribed node(s) outside the valid domain."
                        )

            else:
                errors.append(
                    f"ERROR  — 'bc_mask' dtype must be bool or float, "
                    f"got {arr.dtype}. "
                    "Use a boolean array (pressure computed from velocity) or "
                    "a float array with NaN at interior nodes (pressure "
                    "prescribed directly)."
                )

    # 5. model-specific fields ------------------------------------------------

    if model is not None:

        if model not in _MODEL_REQUIRED:
            errors.append(
                f"ERROR  — Unknown model '{model}'. "
                f"Valid options: {sorted(_MODEL_REQUIRED)}."
            )
        else:
            required_2d = _MODEL_REQUIRED[model]
            info.append(
                f"INFO   — Checking required fields for "
                f"model='{model}': {sorted(required_2d)}."
            )
            for key in sorted(required_2d):
                field_ok = check_2d(key, required=True)
                if field_ok and check_nan_consistency:
                    _check_nan_consistency(
                        key, data[key], data.get('domain'), warnings=warnings
                    )
    else:
        for key in sorted(_2D_FIELDS - {'domain', 'bc_mask'}):
            if key in data:
                field_ok = check_2d(key, required=False)
                if field_ok and check_nan_consistency:
                    _check_nan_consistency(
                        key, data[key], data.get('domain'), warnings=warnings
                    )

    # 6. variance non-negativity ----------------------------------------------

    for key in ('u_var', 'v_var'):
        if key in data and isinstance(data[key], np.ndarray):
            arr = data[key]
            domain = data.get('domain')
            valid = ~np.isnan(arr)
            if domain is not None and isinstance(domain, np.ndarray):
                valid &= domain
            if valid.any() and np.any(arr[valid] < 0):
                errors.append(
                    f"ERROR  — '{key}' contains negative values "
                    "(variance must be ≥ 0)."
                )

    # 7. sentinel-zero check --------------------------------------------------

    if domain_ok:
        dom = data['domain']
        for key in _2D_FIELDS - {'domain', 'bc_mask'}:
            if key in data and isinstance(data[key], np.ndarray):
                arr = data[key]
                if arr.shape == dom.shape:
                    zero_inside = (arr == 0) & dom
                    if zero_inside.any():
                        warnings.append(
                            f"WARNING — '{key}' has {int(zero_inside.sum())} "
                            "zero(s) inside the valid domain — ensure raw "
                            "sentinel zeros have been replaced with NaN."
                        )

    # 8. scalars --------------------------------------------------------------

    for key in ('rho', 'nu'):
        if key not in data:
            errors.append(f"ERROR  — Missing required scalar '{key}'.")
        else:
            _check_scalar(key, data[key], positive=True,
                          errors=errors, ok=ok)

    # q_0: required only when bc_mask is boolean
    if bc_mask_mode == 'boolean':
        if 'q_0' not in data or data['q_0'] is None:
            warnings.append(
                "WARNING — 'q_0' not present; reference pressure will be "
                "computed as the average over all Dirichlet BC nodes at runtime."
            )
        else:
            _check_scalar('q_0', data['q_0'], positive=True,
                          errors=errors, ok=ok)

    elif bc_mask_mode == 'numeric':
        if 'q_0' in data and data['q_0'] is not None:
            warnings.append(
                "WARNING — 'q_0' is set but will be ignored because "
                "'bc_mask' is numeric (pressure prescribed directly)."
            )
        else:
            info.append(
                "INFO   — 'q_0' not required; "
                "'bc_mask' is numeric (pressure prescribed directly)."
            )

    else:
        if 'q_0' in data and data['q_0'] is not None:
            _check_scalar('q_0', data['q_0'], positive=True,
                          errors=errors, ok=ok)
        else:
            warnings.append(
                "WARNING — 'q_0' not present; "
                "will be computed from Dirichlet BC nodes at runtime."
            )

    # 9. unknown keys ---------------------------------------------------------

    unknown = set(data.keys()) - _ALL_KNOWN_KEYS
    for key in sorted(unknown):
        msg = (
            f"ERROR  — Unexpected key '{key}' is not part of the specification."
            if strict else
            f"WARNING — Unexpected key '{key}' is not part of the specification."
        )
        (errors if strict else warnings).append(msg)

    # 10. final assembly ------------------------------------------------------

    valid = len(errors) == 0

    result_line = (
        "RESULT — Data format is VALID ✓"
        + (f" (model='{model}')" if model else "")
        if valid else
        f"RESULT — Data format is INVALID ✗ ({len(errors)} error(s) found)"
    )

    messages = [result_line] + errors + warnings + info + ok

    return valid, messages


# internal helpers ------------------------------------------------------------

def _check_scalar(
    key: str,
    val,
    *,
    positive: bool,
    errors: list[str],
    ok: list[str],
) -> None:
    """Validate a scalar field."""

    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            val = val.item()
        else:
            errors.append(
                f"ERROR  — '{key}' must be a scalar, "
                f"got array with shape={val.shape}."
            )
            return

    try:
        fval = float(val)
    except (TypeError, ValueError):
        errors.append(
            f"ERROR  — '{key}' could not be cast to float "
            f"(got {type(val).__name__})."
        )
        return

    if not np.isfinite(fval):
        errors.append(f"ERROR  — '{key}' must be finite, got {fval}.")
        return

    if positive and fval <= 0:
        errors.append(f"ERROR  — '{key}' must be > 0, got {fval}.")
        return

    ok.append(f"OK     — '{key}' = {fval}.")


def _check_nan_consistency(
    key: str,
    arr: np.ndarray,
    domain: np.ndarray | None,
    *,
    warnings: list[str],
) -> None:
    """Warn about NaN holes inside the domain or leakage outside it."""

    if domain is None or not isinstance(domain, np.ndarray):
        return
    if domain.dtype != bool or domain.shape != arr.shape:
        return

    nan_mask = np.isnan(arr)

    holes = nan_mask & domain
    if holes.any():
        warnings.append(
            f"WARNING — '{key}' has {int(holes.sum())} "
            "NaN value(s) inside the valid domain."
        )

    leakage = ~nan_mask & ~domain
    if leakage.any():
        warnings.append(
            f"WARNING — '{key}' has {int(leakage.sum())} "
            "non-NaN value(s) outside the domain mask."
        )


# pretty printer --------------------------------------------------------------

def report(
    data: dict,
    model: Literal['NS', 'TH', 'RANS'] | None = None,
    *,
    strict: bool = False,
    exit_on_error: bool = False,
) -> bool:
    """
    Validate *data* and pretty-print the report with ANSI colours.

    Parameters
    ----------
    exit_on_error : bool
        If True, calls sys.exit when validation fails.
    """

    valid, messages = check_data_format(data, model=model, strict=strict)

    _COLOURS = {
        'RESULT' : '\033[1m',
        'ERROR'  : '\033[38;5;203m',
        'WARNING': '\033[38;5;221m',
        'OK'     : '\033[38;5;114m',
        'INFO'   : '\033[38;5;153m',
    }
    _RESET = '\033[0m'

    print()
    for msg in messages:
        prefix = msg.split('—', 1)[0].strip()
        print(f"  {_COLOURS.get(prefix, '')}{msg}{_RESET}")
    print()

    if exit_on_error and not valid:
        sys.exit('Invalid dataset')

    return valid