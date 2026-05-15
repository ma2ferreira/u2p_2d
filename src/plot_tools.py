#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author:     Manuel Ferreira
Created:    April 2026
Updated:    May 2026
Version:    v1.1

Description:
    Visualisation tools for 2D flow fields. Implements Line Integral
    Convolution (LIC) for texture-based vector field rendering, blended with
    a diverging pressure colormap to produce publication-quality flow images.

Includes:
    plot_lic_pressure   High-level LIC + pressure visualisation.
    plot_residuals      Visualisation of residual fields.

Internal:
    _norm
    _lic
    _lic_pressure
    _make_axes
    _add_pressure_colorbar
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator

__all__ = ['plot_lic_pressure', 'plot_residuals']

# global matplotlib style
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
    'figure.dpi': 300,
})

def plot_lic_pressure(x, y, u, v, p, t, dt, cmap='RdBu_r'):
    '''
    Display a coloured LIC image with an accurate pressure colorbar.
    PARAMETERS
    x, y : ndarray
        1D coordinate arrays defining the structured grid.
    u, v : ndarray (2D)
        Velocity components on the grid.
    p : ndarray (2D)
        Pressure field used for colormap blending.
    t : float
        Integration length for LIC stream tracing.
    dt : float
        Time step for numerical integration.
    cmap : str, optional
        Matplotlib colormap for pressure overlay. Default is 'RdBu_r'.
     
    RETURNS
    fig : matplotlib.figure.Figure
        Generated figure object.
    ax : matplotlib.axes.Axes
        Main plotting axes.
    cb : matplotlib.colorbar.Colorbar
        Pressure colorbar.
   '''

    lic = _lic(x, y, u, v, t, dt)
    lic_img = _lic_pressure(lic, p, cmap=cmap)

    AR = np.ptp(y) / (np.ptp(x) + 1e-12)

    fig, ax = _make_axes(
        wf=80,
        margins=[15, 15, 15, 5],
        xlim=(np.min(x), np.max(x)),
        ylim=(np.min(y), np.max(y)),
        xlabel=r'$x$',
        ylabel=r'$y$',
        AR=AR
    )

    extent = [np.min(x), np.max(x), np.min(y), np.max(y)]

    ax.imshow(
        lic_img,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
    )

    cb = _add_pressure_colorbar(fig, ax, p, cmap)

    return fig, ax, cb


def plot_residuals(r, x, y, cmap='viridis'):
    '''
    Visualise a scalar residual field on a structured 2D grid.

    PARAMETERS
    r : ndarray (2D)
        Residual field to be visualised.
    x, y : ndarray
        1D coordinate arrays defining the structured grid.
    cmap : str, optional
        Colormap used for rendering the residual field. Default is 'viridis'.

    RETURNS
    fig : matplotlib.figure.Figure
        Generated figure object.
    ax : matplotlib.axes.Axes
        Main plotting axes.
    cb : matplotlib.colorbar.Colorbar
        Colorbar associated with the residual field.
    '''

    AR = np.ptp(y) / (np.ptp(x) + 1e-12)

    fig, ax = _make_axes(
        wf=80,
        margins=[15, 15, 15, 5],
        xlim=(np.min(x), np.max(x)),
        ylim=(np.min(y), np.max(y)),
        xlabel=r'$x$',
        ylabel=r'$y$',
        AR=AR
    )

    extent = [np.min(x), np.max(x), np.min(y), np.max(y)]

    im = ax.imshow(
        r,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
        cmap=cmap
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$r_0^*$')

    return fig, ax, cb

def _norm(f):
    '''
    Normalise array using robust min-max scaling with NaN and inf handling.

    PARAMETERS
    f : ndarray
        Input scalar field.

    RETURNS
    ndarray
        Normalised field in the range [0, 1].
    '''

    f = np.asarray(f)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    fmin = np.min(f)
    fmax = np.max(f)

    return (f - fmin) / (fmax - fmin + 1e-12)


def _lic(x, y, u, v, t, dt):
    '''
    Compute a Line Integral Convolution (LIC) texture from a 2D vector field.

    PARAMETERS
    x, y : ndarray
        1D coordinate arrays defining the structured grid.
    u, v : ndarray (2D)
        Velocity components on the grid.
    t : float
        Integration length for streamline tracing.
    dt : float
        Integration time step.

    RETURNS
    ndarray (2D)
        LIC texture aligned with the input velocity field.
    '''

    noise = np.random.rand(*u.shape) - 0.5

    u_itp = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=np.nan)
    v_itp = RegularGridInterpolator((y, x), v, bounds_error=False, fill_value=np.nan)
    n_itp = RegularGridInterpolator((y, x), noise, bounds_error=False, fill_value=0.0)

    xi0 = np.tile(x, len(y))
    yi0 = np.repeat(y, len(x))

    def trace(sign):
        xi, yi = xi0.copy(), yi0.copy()
        acc = np.zeros_like(xi)
        cnt = np.zeros_like(xi)
        active = np.ones_like(xi, dtype=bool)

        steps = int(t / dt)

        for _ in range(steps):
            pos = np.stack([yi, xi], axis=1)

            u_val = u_itp(pos)
            v_val = v_itp(pos)

            valid = np.isfinite(u_val) & np.isfinite(v_val) & active
            active &= valid

            if not np.any(active):
                break

            mag = np.sqrt(u_val[valid]**2 + v_val[valid]**2) + 1e-8
            mag = mag**0.5

            u_val = u_val[valid] / mag
            v_val = v_val[valid] / mag

            xi[valid] += sign * u_val * dt
            yi[valid] += sign * v_val * dt

            pos2 = np.stack([yi, xi], axis=1)
            acc[valid] += n_itp(pos2)[valid]
            cnt[valid] += 1

        return acc, cnt

    af, cf = trace(+1)
    ab, cb = trace(-1)

    acc = af + ab
    cnt = cf + cb

    out = np.zeros_like(acc)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask]

    return out.reshape(u.shape)


def _lic_pressure(lic, p, cmap='RdBu_r'):
    '''
    Blend a LIC texture with a pressure field using a diverging colormap.

    PARAMETERS
    lic : ndarray (2D)
        LIC texture.
    p : ndarray (2D)
        Pressure field.
    cmap : str, optional
        Matplotlib colormap for pressure mapping. Default is 'RdBu_r'.

    RETURNS
    ndarray (2D, RGB)
        RGB image combining LIC structure and pressure shading.
    '''

    p = np.asarray(p)

    p_f = p[np.isfinite(p)]
    vmax = np.max(np.abs(p_f)) + 1e-12

    pn = np.clip((p + vmax) / (2 * vmax), 0, 1)

    lic_n = _norm(lic)

    color = plt.cm.get_cmap(cmap)(pn)[..., :3]

    brightness = 0.4 + 0.6 * lic_n

    img = color * brightness[..., None]
    img = np.where(img == 0, 1.0, img)

    return np.clip(img, 0, 1)


def _make_axes(wf, margins, xlim, ylim, xlabel, ylabel, *, AR=0.618,
               xscale='linear', yscale='linear', title=None, grid=False, **kwargs):
    '''
    Create a publication-quality Matplotlib figure and axes with controlled
    aspect ratio and margins.

    PARAMETERS
    wf : float
        Figure width (mm).
    margins : list
        [left, right, bottom, top] margins in mm.
    xlim, ylim : tuple
        Axis limits.
    xlabel, ylabel : str
        Axis labels.
    AR : float, optional
        Aspect ratio (height/width). Default is 0.618.
    xscale, yscale : str, optional
        Axis scaling ('linear', 'log', etc.).
    title : str, optional
        Plot title.
    grid : bool, optional
        Whether to display grid lines.
    **kwargs :
        Additional axis styling parameters.

    RETURNS
    fig : matplotlib.figure.Figure
        Created figure.
    ax : matplotlib.axes.Axes
        Configured axes.
    '''

    l, r, b, t = margins

    wf = float(wf)
    AR = float(AR)

    wa = wf - l - r
    ha = AR * wa
    hf = ha + b + t

    fig, ax = plt.subplots(
        figsize=(wf / 25.4, hf / 25.4),
        dpi=300
    )

    fig.subplots_adjust(
        left=l / wf,
        right=1 - r / wf,
        bottom=b / hf,
        top=1 - t / hf
    )

    ax.set(
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        xscale=xscale,
        yscale=yscale,
        axisbelow=True,
        **kwargs
    )

    if grid:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    if title:
        ax.set_title(title)

    return fig, ax


def _add_pressure_colorbar(fig, ax, p, cmap):
    '''
    Add a symmetric diverging colorbar for a pressure field.

    PARAMETERS
    fig : matplotlib.figure.Figure
        Figure containing the axes.
    ax : matplotlib.axes.Axes
        Target axes for positioning the colorbar.
    p : ndarray (2D)
        Pressure field used to determine color scale limits.
    cmap : str
        Colormap used for pressure mapping.

    RETURNS
    cb : matplotlib.colorbar.Colorbar
        Configured colorbar object.
    '''

    p = np.asarray(p)
    vmax = np.max(np.abs(p[np.isfinite(p)])) + 1e-12

    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    bbox = ax.get_position()

    cax = fig.add_axes([
        bbox.x1 + 0.02,
        bbox.y0,
        0.02,
        bbox.height * 0.8
    ])

    cb = fig.colorbar(sm, cax=cax)
    cb.ax.set_title('$p$ (Pa)', pad=10, loc='left')
    
    return cb