#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:    May 2026
Version:    v1.1

Structured data validation module.
'''

from __future__ import annotations
import numpy as np
from typing import Literal
import sys


# field specifications --------------------------------------------------------

# keys that are always required regardless of model
_ALWAYS_REQUIRED = {'x', 'y', 'domain', 'rho', 'nu'}

# keys required per model
_MODEL_REQUIRED: dict[str, set[str]] = {
    'NS': {'u', 'v', 'u_t', 'v_t'},
    'TH': {'u', 'v', 'u_mean', 'v_mean'},
    'RANS': {'u_mean', 'v_mean', 'u_var', 'v_var', 'uv_mean'},
}

# every 2-D field key that may appear
_2D_FIELDS = {
    'domain',
    'bc_mask',
    'u',
    'v',
    'u_t',
    'v_t',
    'u_mean',
    'v_mean',
    'u_var',
    'v_var',
    'uv_mean',
}

# scalar keys
_SCALAR_KEYS = {'rho', 'nu', 'q_0'}

# all known keys
_ALL_KNOWN_KEYS = {'x', 'y'} | _2D_FIELDS | _SCALAR_KEYS


# public API ------------------------------------------------------------------

def check_data_format(
    data: dict,
    model: Literal['NS', 'TH', 'RANS'] | None = None,
    *,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    '''
    Validate *data* against the structured-grid data format requirements.

    PARAMETERS
    data : dict
        The data dictionary to validate.

    model : {'NS', 'TH', 'RANS'} or None
        When given, checks that all fields required by that model are
        present and valid.

    strict : bool
        If True, unknown keys are treated as errors.
        If False, unknown keys produce warnings only.

    RETURNS
    valid : bool
        True if no errors were found.

    messages : list[str]
        Collected validation messages.
    '''

    if not isinstance(data, dict):
        return (
            False,
            [
                f'ERROR  — \'data\' must be a dict, '
                f'got {type(data).__name__}.'
            ],
        )

    errors: list[str] = []
    warnings: list[str] = []
    ok: list[str] = []
    info: list[str] = []

    # 1. determine grid dimensions from x / y ---------------------------------

    nx = ny = None

    for axis, key in (('x', 'x'), ('y', 'y')):

        if key not in data:
            errors.append(
                f'ERROR  — Missing required coordinate array \'{key}\'.'
            )
            continue

        arr = data[key]

        if not isinstance(arr, np.ndarray):
            errors.append(
                f'ERROR  — \'{key}\' must be a NumPy array, '
                f'got {type(arr).__name__}.'
            )
            continue

        if arr.ndim != 1:
            errors.append(
                f'ERROR  — \'{key}\' must be 1-D, got ndim={arr.ndim}.'
            )
            continue

        if not np.issubdtype(arr.dtype, np.floating):
            warnings.append(
                f'WARNING — \'{key}\' dtype is {arr.dtype}; '
                'expected a floating-point type.'
            )

        if len(arr) < 2:
            errors.append(
                f'ERROR  — \'{key}\' must have length ≥ 2, '
                f'got {len(arr)}.'
            )
            continue

        # Strictly increasing
        if not np.all(np.diff(arr) > 0):
            errors.append(
                f'ERROR  — \'{key}\' must be strictly increasing '
                '(left→right / top→bottom convention).'
            )

        if axis == 'x':
            nx = len(arr)
            ok.append(
                f'OK     — \'x\' is a valid 1-D coordinate array, nx={nx}.'
            )
        else:
            ny = len(arr)
            ok.append(
                f'OK     — \'y\' is a valid 1-D coordinate array, ny={ny}.'
            )

    # 2. 2-D validation helper ------------------------------------------------

    def check_2d(key: str, *, required: bool) -> bool:
        '''
        Return True if field passes all 2-D checks.
        '''

        if key not in data:

            if required:
                errors.append(
                    f'ERROR  — Missing required 2-D field \'{key}\'.'
                )

            return False

        arr = data[key]

        if not isinstance(arr, np.ndarray):
            errors.append(
                f'ERROR  — \'{key}\' must be a NumPy array, '
                f'got {type(arr).__name__}.'
            )
            return False

        if arr.ndim != 2:
            errors.append(
                f'ERROR  — \'{key}\' must be 2-D, got ndim={arr.ndim}.'
            )
            return False

        # Shape check
        if ny is not None and nx is not None:

            if arr.shape != (ny, nx):
                errors.append(
                    f'ERROR  — \'{key}\' shape {arr.shape} does not match '
                    f'(ny={ny}, nx={nx}).'
                )
                return False

        ok.append(
            f'OK     — \'{key}\' is a valid 2-D array, '
            f'shape={arr.shape}, dtype={arr.dtype}.'
        )

        return True

    # 3. domain mask ----------------------------------------------------------

    domain_ok = check_2d('domain', required=True)

    if domain_ok:

        mask = data['domain']

        if mask.dtype != bool:

            errors.append(
                f'ERROR  — \'domain\' must have dtype bool, '
                f'got {mask.dtype}. '
                'Cast with: '
                'data[\'domain\'] = data[\'domain\'].astype(bool)'
            )

        else:

            n_valid = int(mask.sum())
            n_total = mask.size

            ok[-1] += (
                f', valid nodes={n_valid}/{n_total}.'
            )

    # 4. bc_mask --------------------------------------------------------------

    if 'bc_mask' in data:

        bc_ok = check_2d('bc_mask', required=False)

        if bc_ok and data['bc_mask'].dtype != bool:

            errors.append(
                f"ERROR  — 'bc_mask' must have dtype bool, "
                f"got {data['bc_mask'].dtype}."
            )

    else:

        warnings.append(
            'WARNING — \'bc_mask\' not present; '
            'Dirichlet BC nodes will not be explicitly set.'
        )

    # 5. model-specific required fields ---------------------------------------

    if model is not None:

        if model not in _MODEL_REQUIRED:

            errors.append(
                f'ERROR  — Unknown model \'{model}\'. '
                f'Valid options: {list(_MODEL_REQUIRED)}.'
            )

        else:

            required_2d = _MODEL_REQUIRED[model]

            info.append(
                f'INFO   — Checking required fields for '
                f'model=\'{model}\': {sorted(required_2d)}.'
            )

            for key in sorted(required_2d):

                field_ok = check_2d(key, required=True)

                if field_ok:

                    _check_nan_consistency(
                        key,
                        data[key],
                        data.get('domain'),
                        warnings=warnings,
                    )

    else:

        # validate any present optional 2-D fields
        for key in sorted(_2D_FIELDS - {'domain', 'bc_mask'}):

            if key in data:

                field_ok = check_2d(key, required=False)

                if field_ok:

                    _check_nan_consistency(
                        key,
                        data[key],
                        data.get('domain'),
                        warnings=warnings,
                    )

    # 6. Variance fields ------------------------------------------------------

    for key in ('u_var', 'v_var'):

        if key in data and isinstance(data[key], np.ndarray):

            arr = data[key]
            domain = data.get('domain')

            valid_mask = ~np.isnan(arr)

            if domain is not None and isinstance(domain, np.ndarray):
                valid_mask &= domain

            if valid_mask.any() and np.any(arr[valid_mask] < 0):

                errors.append(
                    f'ERROR  — \'{key}\' contains negative values '
                    '(variance must be ≥ 0).'
                )

    # 7. Scalar fields --------------------------------------------------------

    for key in ('rho', 'nu'):

        if key not in data:
            errors.append(
                f'ERROR  — Missing required scalar \'{key}\'.'
            )
            continue

        _check_scalar(
            key,
            data[key],
            positive=True,
            errors=errors,
            ok=ok,
        )

    if 'q_0' in data and data['q_0'] is not None:

        _check_scalar(
            'q_0',
            data['q_0'],
            positive=True,
            errors=errors,
            ok=ok,
        )

    elif 'q_0' not in data:

        warnings.append(
            'WARNING — \'q_0\' not present; '
            'will be computed from Dirichlet BC nodes at runtime.'
        )

    else:

        ok.append(
            'OK     — \'q_0\' is None; '
            'reference pressure will be computed from Dirichlet BC nodes.'
        )
        
    # 8. unknown keys ---------------------------------------------------------

    unknown = set(data.keys()) - _ALL_KNOWN_KEYS

    for key in sorted(unknown):

        msg = (
            f'ERROR  — Unexpected key \'{key}\' '
            'is not part of the specification.'
            if strict
            else
            f'WARNING — Unexpected key \'{key}\' '
            'is not part of the specification.'
        )

        (errors if strict else warnings).append(msg)

    # 9. final assembly -------------------------------------------------------

    messages: list[str] = (
        errors + warnings + info + ok
    )

    valid = len(errors) == 0

    if valid:

        messages.insert(
            0,
            'RESULT — Data format is VALID ✓'
            + (f' (model=\'{model}\')' if model else '')
        )

    else:

        messages.insert(
            0,
            f'RESULT — Data format is INVALID ✗ '
            f'({len(errors)} error(s) found)'
        )

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
    '''
    Validate scalar value.
    '''

    if isinstance(val, np.ndarray):

        if val.ndim == 0:
            val = val.item()

        else:
            errors.append(
                f'ERROR  — \'{key}\' must be a scalar, '
                f'got array with shape={val.shape}.'
            )
            return

    try:
        fval = float(val)

    except (TypeError, ValueError):

        errors.append(
            f'ERROR  — \'{key}\' could not be cast to float '
            f'(got {type(val).__name__}).'
        )
        return

    if not np.isfinite(fval):

        errors.append(
            f'ERROR  — \'{key}\' must be finite, got {fval}.'
        )
        return

    if positive and fval <= 0:

        errors.append(
            f'ERROR  — \'{key}\' must be > 0, got {fval}.'
        )
        return

    ok.append(f'OK     — \'{key}\' = {fval}.')


def _check_nan_consistency(
    key: str,
    arr: np.ndarray,
    domain: np.ndarray | None,
    *,
    warnings: list[str],
) -> None:
    '''
    Check NaN consistency with domain mask.
    '''

    if domain is None:
        return

    if not isinstance(domain, np.ndarray):
        return

    if domain.dtype != bool:
        return

    if domain.shape != arr.shape:
        return

    nan_mask = np.isnan(arr)

    # NaNs inside valid domain
    holes = nan_mask & domain

    if holes.any():

        warnings.append(
            f'WARNING — \'{key}\' has {int(holes.sum())} '
            'NaN value(s) inside the valid domain mask.'
        )

    # non-NaNs outside domain
    leakage = ~nan_mask & ~domain

    if leakage.any():

        warnings.append(
            f'WARNING — \'{key}\' has {int(leakage.sum())} '
            'non-NaN value(s) outside the domain mask.'
        )

# pretty printer --------------------------------------------------------------

def report(
    data: dict,
    model: Literal['NS', 'TH', 'RANS'] | None = None,
    *,
    strict: bool = False,
) -> bool:
    '''
    Pretty-print validation report.
    '''

    valid, messages = check_data_format(
        data,
        model=model,
        strict=strict,
    )

    # ANSI colours
    _COLOURS = {
        'RESULT' : '\033[1m',           # bold (keep emphasis instead of color)
        'ERROR' : '\033[38;5;203m',     # soft pastel red / coral
        'WARNING': '\033[38;5;221m',    # soft pastel yellow
        'OK' : '\033[38;5;114m',        # soft pastel green
        'INFO' : '\033[38;5;153m',      # soft pastel blue
    }

    _RESET = '\033[0m'

    print()

    for msg in messages:

        prefix = msg.split('—', 1)[0].strip()

        colour = _COLOURS.get(prefix, '')

        print(f'  {colour}{msg}{_RESET}')

    print()
    
    if not valid:
        sys.exit('Invalid dataset')

    return valid