"""Dual KAN + reference PRE-COLLISION fneq + DISCRETE D2Q9 MRT-LBM inverse.

Only the mesoscopic MRT-LBM collision/streaming equation is used as a
GOVERNING-EQUATION RESIDUAL; NO macroscopic Navier-Stokes momentum equations,
NO extra pressure output, and NO pressure-gradient constraint are included.

Two KAN branches:
  - macro: (x,y,t) -> (rho, u, v)
  - kinetic: (x,y,t) -> 9 pre-collision f_i^neq
The macro and kinetic labels are sampled from the same reference points.

D2Q9 periodic one-step residual:
    f_i((x+c_ix/N) mod 1, (y+c_iy/N) mod 1, t+U0/N)
      - f_i(x,y,t) + [M^{-1} S(nu) M f_neq(x,y,t)]_i = 0.
The macro input velocity is nondimensional; in the lattice equilibrium its
velocity is multiplied by U0. The loss scales the discrete residual by N.
The globally trainable viscosity enters S(nu) through the two shear rates.

DEFAULT: full-time inverse problem, positive viscosity initialized to 0.002.
The specified representative time snapshots are withheld from ALL labeled
supervision (rho,u,v, and 9 fneq populations), but the unlabeled MRT residual
is sampled throughout the time domain, including those snapshot times. Thus
this tests physics-informed temporal interpolation, NOT time extrapolation.

Data: NPZ with pre-collision fneq[T,9,N,N] generated from the reference MRT
solver. Supervised fneq helps reduce kinetic branch freedom but does NOT
establish unique viscosity identification. Resolution and internal consistency
do not, by themselves, prove high fidelity or mesh independence.

Single-file plots are launched as a subprocess of THIS SAME .py module with
--plot-only, before importing torch, reducing Windows OpenMP conflicts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Embedded visualization functions. This SINGLE .py file also serves as its
# own standalone plotter. When launched with --plot-only, the plotter runs
# BEFORE importing PyTorch, which reduces Windows/OpenMP DLL conflicts.
# ----------------------------------------------------------------------------
def _plot_args_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plot-only', action='store_true', help='Plot previously evaluated results without importing torch')
    p.add_argument('--output', type=Path, required=True, help='Folder containing test_fields.npz')
    p.add_argument('--dpi', type=int, default=220, help='Resolution of saved PNG images')
    return p.parse_args()


def save(fig, dest, dpi):
    fig.savefig(dest, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[FIGURE]', dest, flush=True)


def time_name(t):
    return ('t' + f'{float(t):.3f}'.replace('.', 'p')).replace('-', 'm')


def add_image(ax, arr, extent, name, cmap, low=None, high=None):
    im = ax.imshow(arr, origin='lower', extent=extent, interpolation='nearest',
                   cmap=cmap, vmin=low, vmax=high, aspect='equal')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.035)


def triptych(reference, prediction, extent, title, filename, dpi, *, cmap='RdBu_r', signed=True):
    diff = np.abs(prediction - reference)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    if signed:
        vmax = max(float(np.max(np.abs(reference))), float(np.max(np.abs(prediction))), 1e-12)
        vmin, vmax = -vmax, vmax
    else:
        vmin = min(float(np.min(reference)), float(np.min(prediction)))
        vmax = max(float(np.max(reference)), float(np.max(prediction)))
        if vmin == vmax: vmax += 1e-10
    add_image(axes[0], reference, extent, 'Reference', cmap, vmin, vmax)
    add_image(axes[1], prediction, extent, 'Dual KAN (MRT+fneq) prediction', cmap, vmin, vmax)
    add_image(axes[2], diff, extent, 'Absolute error', 'magma', 0, max(float(diff.max()), 1e-12))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    save(fig, filename, dpi)


def field_grid(truth, pred, extent, t, path, dpi):
    # Side-by-side comparison across u,v and speed. Use consistent scales for
    # reference and prediction to avoid visually misleading model comparisons.
    speed_true = np.sqrt(truth[..., 1]**2 + truth[..., 2]**2)
    speed_pred = np.sqrt(pred[..., 1]**2 + pred[..., 2]**2)
    labels = [('u (streamwise)', truth[..., 1], pred[..., 1], True),
              ('v (cross-stream)', truth[..., 2], pred[..., 2], True),
              ('Speed |U|', speed_true, speed_pred, False)]
    fig, axes = plt.subplots(3, 3, figsize=(12.7, 11.1), constrained_layout=True)
    for i, (name, a, b, signed) in enumerate(labels):
        err = np.abs(a-b) if i < 2 else np.sqrt((truth[...,1]-pred[...,1])**2+(truth[...,2]-pred[...,2])**2)
        vmax = max(float(np.max(np.abs(a))),float(np.max(np.abs(b))),1e-12) if signed else max(float(np.max(a)),float(np.max(b)),1e-12)
        low = -vmax if signed else 0
        cols = [(a, 'Reference', 'RdBu_r' if signed else 'viridis', low, vmax),
                (b, 'Dual KAN (MRT+fneq)', 'RdBu_r' if signed else 'viridis', low, vmax),
                (err, '|error|' if i<2 else 'Vector error', 'magma', 0, max(float(err.max()),1e-12))]
        for j, (arr, label, cmap, vmin, vmax) in enumerate(cols):
            add_image(axes[i,j], arr, extent, f'{name} - {label}', cmap, vmin, vmax)
    fig.suptitle(f'2D double-shear layer, t = {t:.2f} | full-field velocity reconstruction', fontweight='bold', fontsize=14)
    save(fig, path, dpi)


def training_curves(out, figs, dpi):
    path = out/'history.csv'
    if not path.is_file():
        return
    with path.open('r', newline='', encoding='utf-8-sig') as fp:
        records = list(csv.DictReader(fp))
    if not records:
        return
    def col(name):
        return np.array([float(r.get(name) or 'nan') for r in records], dtype=float)
    x = col('global_epoch') + 1
    phases = [r.get('phase','') for r in records]
    physics_start = next((int(x[i]) for i, p in enumerate(phases) if p == 'physics'), None)
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.3), constrained_layout=True)
    ax = axs[0,0]
    for name, label in [('total_loss', 'Total loss'),('data_loss','Data loss'),('ic_loss','Initial-condition loss')]:
        y = col(name)
        ax.semilogy(x, np.maximum(y,1e-14), label=label, linewidth=1.1)
    ax.set_title('Training losses'); ax.legend(fontsize=8); ax.set_ylabel('Loss')
    ax = axs[0,1]
    y = col('mrt_loss_scaled')
    active = np.isfinite(y) & (y > 0)
    if active.any(): ax.semilogy(x[active], y[active], label='Discrete MRT residual', linewidth=1.2)
    fs = col('fneq_observation_loss')
    active_fs = np.isfinite(fs) & (fs > 0)
    if active_fs.any(): ax.semilogy(x[active_fs], fs[active_fs], label='Supervised fneq loss', linewidth=1.1)
    ax.set_title('Physics residuals'); ax.set_ylabel('Unweighted MSE'); ax.legend(fontsize=8)
    ax = axs[1,0]
    vals = col('validation_loss')
    mask = np.isfinite(vals)
    if mask.any(): ax.semilogy(x[mask], np.maximum(vals[mask],1e-14), label='Held-out supervised validation')
    ax.set_title('Validation error'); ax.set_ylabel('Scaled MSE'); ax.legend(fontsize=8)
    ax = axs[1,1]
    nu = col('nu_nondim')
    if np.isfinite(nu).any():
        ax.plot(x, nu, linewidth=1.3, label='Model viscosity')
        summary = out/'run_summary.json'
        if summary.is_file():
            info = json.loads(summary.read_text(encoding='utf-8'))
            actual = info.get('reference_nu_nondim_FOR_REPORT_ONLY')
            if actual is not None: ax.axhline(float(actual), linestyle='--', linewidth=1.0, label='Reference viscosity')
    ax.set_title('Global viscosity trajectory'); ax.set_ylabel('Nondimensional nu'); ax.legend(fontsize=8)
    for a in axs.flat:
        a.set_xlabel('Global optimization epoch')
        a.grid(alpha=.22)
        if physics_start is not None: a.axvline(physics_start, linestyle=':', alpha=.55, color='gray')
    fig.suptitle('Dual KAN | supervised fneq + discrete MRT inverse', fontweight='bold', fontsize=13)
    save(fig, figs/'training_curves.png', dpi)
    nu_grad = col('nu_log_grad_pre_clip')
    if np.isfinite(nu_grad).any():
        fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
        mask = np.isfinite(nu_grad)
        ax.semilogy(x[mask], np.maximum(np.abs(nu_grad[mask]), 1e-15), label='|dL/d(log nu)| before gradient clipping')
        ax.set_xlabel('Global epoch')
        ax.set_ylabel('Absolute viscosity gradient')
        ax.set_title('Viscosity-parameter gradient diagnostic')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        save(fig, figs/'nu_gradient_diagnostics.png', dpi)


def summary_errors(out, figs, dpi):
    f = out/'test_metrics.csv'
    if not f.is_file(): return
    with f.open('r', encoding='utf-8-sig', newline='') as h:
        rec = list(csv.DictReader(h))
    if not rec: return
    t = np.array([float(r['t']) for r in rec])
    fig, ax = plt.subplots(figsize=(8.3, 4.6), constrained_layout=True)
    for key, name in [('u_rel_l2','u'),('v_rel_l2','v'),('vorticity_rel_l2','vorticity')]:
        ax.plot(t,[100*float(r[key]) for r in rec], marker='o', label=name)
    ax.set_title('Held-out test error by physical time')
    ax.set_xlabel('Time t'); ax.set_ylabel('Relative L2 error (%)')
    ax.grid(alpha=.25); ax.legend()
    save(fig, figs/'temporal_error_summary.png', dpi)


def _run_plot_mode():
    args = _plot_args_cli()
    out = args.output.resolve()
    f = out/'test_fields.npz'
    if not f.is_file():
        raise FileNotFoundError(f'Missing {f}. Evaluate model first, then plot.')
    figures = out/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    with np.load(f, allow_pickle=False) as z:
        t = z['time'].copy()
        x = z['x'].copy()
        y = z['y'].copy()
        prediction = z['prediction'].copy()
        truth = z['truth'].copy()
        vort_p = z['vorticity_prediction'].copy()
        vort_t = z['vorticity_truth'].copy()
        e_p = z['spectrum_prediction'].copy()
        e_t = z['spectrum_truth'].copy()
        k = z['spectrum_k'].copy()
    assert prediction.shape == truth.shape, 'predictions and reference shapes mismatch'
    fneq_data = None
    with np.load(f, allow_pickle=False) as z:
        if 'fneq_prediction' in z and 'fneq_truth' in z:
            fneq_data = (z['fneq_prediction'], z['fneq_truth'])
    if prediction.shape[0] != len(t): raise ValueError('time count mismatch')
    extent=[float(x.min()),float(x.max()),float(y.min()),float(y.max())]
    for i, time in enumerate(t):
        stamp=time_name(time)
        field_grid(truth[i],prediction[i],extent,float(time),figures/f'velocity_{stamp}.png',args.dpi)
        triptych(vort_t[i],vort_p[i],extent,
                 f'Vorticity field, t = {float(time):.2f}',
                 figures/f'vorticity_{stamp}.png',args.dpi)
        triptych(truth[i,...,0],prediction[i,...,0],extent,
                 f'Density field, t = {float(time):.2f}',
                 figures/f'density_{stamp}.png',args.dpi,cmap='viridis',signed=False)
        if fneq_data is not None:
            fnpred, fntrue = fneq_data[0][i], fneq_data[1][i]
            fn_mag_pred = np.sqrt(np.sum(fnpred.astype(np.float64)**2,axis=-1))
            fn_mag_true = np.sqrt(np.sum(fntrue.astype(np.float64)**2,axis=-1))
            triptych(fn_mag_true, fn_mag_pred, extent,
                     f'Norm of nine pre-collision fneq populations at t={float(time):.2f}',
                     figures/f'fneq_norm_{stamp}.png',args.dpi,cmap='viridis',signed=False)
            for direction in [1, 5]:
                triptych(fntrue[...,direction],fnpred[...,direction],extent,
                         f'Pre-collision fneq[{direction}] at t={float(time):.2f}',
                         figures/f'fneq_i{direction}_{stamp}.png',args.dpi)
        fig,ax=plt.subplots(figsize=(7.4,4.5),constrained_layout=True)
        nonzero=k>0
        ax.semilogy(k[nonzero],np.maximum(e_t[i][nonzero],1e-20),label='Reference', linewidth=1.4)
        ax.semilogy(k[nonzero],np.maximum(e_p[i][nonzero],1e-20),label='Dual KAN (MRT+fneq)',linewidth=1.4)
        ax.set_xlabel('Radial wavenumber k'); ax.set_ylabel('2D kinetic-energy shell spectrum')
        ax.set_title(f'Energy spectrum at t = {float(time):.2f}')
        ax.grid(alpha=.2); ax.legend()
        save(fig, figures/f'energy_spectrum_{stamp}.png',args.dpi)
    training_curves(out,figures,args.dpi)
    summary_errors(out,figures,args.dpi)
    print('[PLOTS] Created images for',len(t),'test snapshots in',figures,flush=True)



if __name__ == '__main__' and '--plot-only' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    _run_plot_mode()
    raise SystemExit(0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


C_NP = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                 [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)
W_NP = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
M_NP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
    [4, -2, -2, -2, -2, 1, 1, 1, 1],
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, -2, 0, 2, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1],
    [0, 0, -2, 0, 2, 1, 1, -1, -1],
    [0, 1, -1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 1, -1],
], dtype=np.float64)
EXPECTED_MODES = ['rho', 'e', 'eps', 'jx', 'qx', 'jy', 'qy', 'pxx', 'pxy']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KANLinear(nn.Module):
    """Cubic B-spline KAN with an SiLU base branch."""

    def __init__(self, in_features, out_features, grid_size=8, spline_order=3,
                 grid_range=(-1., 1.), scale_noise=0.05):
        super().__init__()
        if grid_size < 2 or spline_order < 1:
            raise ValueError('invalid KAN grid parameters')
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = (torch.arange(-spline_order, grid_size + spline_order + 1)
                 * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer('grid', knots)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(
            out_features, in_features, grid_size + spline_order))
        self.spline_scale = nn.Parameter(torch.ones(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=scale_noise / grid_size)

    def b_splines(self, x):
        xx = x.unsqueeze(-1)
        grid = self.grid
        basis = ((xx >= grid[:, :-1]) & (xx < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            lhs = (xx - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)])
            rhs = (grid[:, k+1:] - xx) / (grid[:, k+1:] - grid[:, 1:-k])
            basis = lhs * basis[:, :, :-1] + rhs * basis[:, :, 1:]
        return basis.contiguous()

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(
            self.b_splines(x).reshape(x.shape[0], -1),
            (self.spline_weight * self.spline_scale.unsqueeze(-1)).reshape(
                self.out_features, -1))
        return base + spline


class KAN(nn.Module):
    def __init__(self, widths, grid_size):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(i, o, grid_size=grid_size)
            for i, o in zip(widths[:-1], widths[1:])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = torch.tanh(x)
        return x


def periodic_features(xyt, t_end):
    x, y, t = xyt[:, :1], xyt[:, 1:2], xyt[:, 2:3]
    return torch.cat([
        torch.sin(2*torch.pi*x), torch.cos(2*torch.pi*x),
        torch.sin(2*torch.pi*y), torch.cos(2*torch.pi*y),
        2*t/t_end-1], dim=1)


class DualKANMRT(nn.Module):
    def __init__(self, n, u0, nu_lattice, t_end, base_rates,
                 width=48, depth=2, grid_size=8, rho_scale=0.01,
                 fneq_scale=0.002, learn_nu=False, nu_init=None):
        super().__init__()
        hidden = [width] * depth
        self.macro = KAN([5] + hidden + [3], grid_size)
        self.kinetic = KAN([5] + hidden + [9], grid_size)
        self.n = float(n)
        self.u0 = float(u0)
        self.t_end = float(t_end)
        self.rho_scale = float(rho_scale)
        self.fneq_scale = float(fneq_scale)
        self.learn_nu = bool(learn_nu)
        if not (self.n > 0 and self.u0 > 0 and self.t_end > 0):
            raise ValueError('n, u0, t_end must be positive')
        if self.learn_nu:
            if nu_init is None or nu_init <= 0:
                raise ValueError('inverse mode requires positive --nu-init (dimensionless)')
            # Convert nondimensional nu to lattice nu without consulting true nu.
            self.log_nu = nn.Parameter(torch.tensor(math.log(nu_init*self.u0*self.n)))
        else:
            if nu_lattice <= 0:
                raise ValueError('reference lattice viscosity must be positive')
            self.register_buffer('fixed_nu', torch.tensor(float(nu_lattice)))

        c = torch.tensor(C_NP, dtype=torch.float32)
        m = torch.tensor(M_NP, dtype=torch.float32)
        inv_m = torch.linalg.inv(m)
        head = np.asarray(base_rates[:7], dtype=np.float32)
        if len(head) != 7:
            raise ValueError('seven MRT non-shear rates are required')
        if not np.allclose(head[[0, 3, 5]], 0):
            raise ValueError('conserved MRT rates must be zero')
        if np.any(head < 0) or np.any(head >= 2):
            raise ValueError('MRT rates must satisfy 0 <= s < 2')
        moments = torch.stack([torch.ones(9), c[:, 0], c[:, 1]])
        proj = torch.eye(9) - moments.T @ torch.linalg.inv(moments @ moments.T) @ moments
        self.register_buffer('c', c)
        self.register_buffer('w', torch.tensor(W_NP, dtype=torch.float32))
        self.register_buffer('m', m)
        self.register_buffer('m_inv', inv_m)
        self.register_buffer('rates_head', torch.tensor(head))
        self.register_buffer('projection', proj)

    @property
    def nu_lattice(self):
        return self.log_nu.exp() if self.learn_nu else self.fixed_nu

    @property
    def nu_nondim(self):
        return self.nu_lattice / (self.n*self.u0)

    def collision_matrix(self):
        s_nu = 1.0 / (0.5 + 3.0*self.nu_lattice)
        rates = torch.cat([self.rates_head, s_nu.expand(2)])
        return self.m_inv @ torch.diag(rates) @ self.m

    def macro_state(self, xyt):
        raw = self.macro(periodic_features(xyt, self.t_end))
        return torch.cat([1.0 + self.rho_scale*raw[:, :1], raw[:, 1:3]], dim=1)

    def fneq(self, xyt):
        raw = self.kinetic(periodic_features(xyt, self.t_end))
        return self.fneq_scale * (raw @ self.projection.T)

    def feq(self, macro):
        rho = macro[:, :1]
        u_latt = self.u0 * macro[:, 1:3]
        cu = u_latt @ self.c.T
        u2 = (u_latt*u_latt).sum(dim=1, keepdim=True)
        return self.w*rho*(1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u2)

    def distributions(self, xyt):
        macro = self.macro_state(xyt)
        eq = self.feq(macro)
        neq = self.fneq(xyt)
        return macro, neq, eq + neq

    def physics_terms(self, xyt):
        """Evaluate D2Q9 MRT collision/streaming residual, periodic x/y."""
        macro, neq, f0 = self.distributions(xyt)
        bsz = xyt.shape[0]
        step = self.u0 / self.n
        xy_next = torch.remainder(xyt[:, None, :2] + self.c[None, :, :]/self.n, 1.0)
        t_next = (xyt[:, None, 2:3] + step).expand(-1, 9, -1)
        shifted = torch.cat([xy_next, t_next], dim=-1).reshape(-1, 3)
        _, _, f_all = self.distributions(shifted)
        f_stream = torch.diagonal(f_all.reshape(bsz, 9, 9), dim1=1, dim2=2)
        collision = neq @ self.collision_matrix().T
        residual = f_stream - f0 + collision
        return {
            'residual': residual,
            'positivity': F.relu(-f0),
            'fneq': neq,
        }


class TrajectoryData:
    """Time-wide supervision except for whole held-out temporal snapshots.

    The test frames must lie strictly inside the interval bracketed by
    supervised training frames, so evaluation is temporal interpolation.
    A held-out frame is excluded from ALL macro observation samples, while
    equation collocation is active throughout [0, t_end].
    """
    def __init__(self, path, test_times=(0.5, 1.0, 1.5, 1.9)):
        with np.load(path, allow_pickle=False) as z:
            self.x = z['x'].astype(np.float32)
            self.y = z['y'].astype(np.float32)
            self.t = z['t'].astype(np.float32)
            self.rho = z['rho'].astype(np.float32)
            self.u = z['u'].astype(np.float32)
            self.v = z['v'].astype(np.float32)
            if 'fneq' not in z.files:
                raise ValueError('Dataset has no fneq! Use dsl_Re1000_N256_MaSafe_with_fneq.npz')
            self.fneq_ref = z['fneq'].astype(np.float32)
            self.ul_ref = z['u_lattice'].astype(np.float32) if 'u_lattice' in z.files else None
            self.vl_ref = z['v_lattice'].astype(np.float32) if 'v_lattice' in z.files else None
            self.diagnostics = json.loads(str(z['diagnostics_json'])) if 'diagnostics_json' in z.files else []
            self.meta = json.loads(str(z['metadata_json']))
        n = int(self.meta['n'])
        shape = (len(self.t), n, n)
        if any(a.shape != shape for a in (self.rho, self.u, self.v)):
            raise ValueError(f'Expected rho,u,v to be {shape}')
        if self.fneq_ref.shape != (len(self.t),9,n,n):
            raise ValueError('fneq must be [T,9,N,N] and match macro data')
        if not all(np.isfinite(a).all() for a in (self.rho,self.u,self.v,self.fneq_ref)):
            raise ValueError('Dataset contains NaN or Inf')
        if self.meta.get('fneq_definition') != 'pre-collision f_i - f_i^eq(rho,u_lattice,v_lattice)':
            raise ValueError('Only pre-collision fneq reference is supported; cannot mix pre/post collision')
        # A few numerical consistency checks; NOT a grid-convergence test.
        check_rng = np.random.default_rng(7843)
        ri = check_rng.integers(len(self.t),size=16000)
        yi = check_rng.integers(n,size=16000)
        xi = check_rng.integers(n,size=16000)
        fk = self.fneq_ref[ri,:,yi,xi].astype(np.float64)
        f_mass = fk.sum(axis=1)
        f_momx = fk @ C_NP[:,0]
        f_momy = fk @ C_NP[:,1]
        self.quality_report = {
            'grid': [n,n], 'snapshots': len(self.t),
            'fneq_shape': list(self.fneq_ref.shape),
            'pre_collision_fneq': True,
            'fneq_sampled_max_abs_mass_moment': float(np.max(np.abs(f_mass))),
            'fneq_sampled_max_abs_momentum_x': float(np.max(np.abs(f_momx))),
            'fneq_sampled_max_abs_momentum_y': float(np.max(np.abs(f_momy))),
            'fneq_global_rms': float(np.sqrt(np.mean(self.fneq_ref.astype(np.float64)**2))),
            'max_recorded_mach': float(max((rec.get('max_mach',0) for rec in self.diagnostics),default=float('nan'))),
            'max_recorded_mass_drift': float(max((abs(rec.get('mass_rel_error',0)) for rec in self.diagnostics),default=float('nan'))),
            'verified_grid_independence': False,
            'limitation': 'Conservation and data consistency audited; grid convergence requires other resolutions.'}
        if any(self.quality_report[k] > 2e-7 for k in (
                'fneq_sampled_max_abs_mass_moment',
                'fneq_sampled_max_abs_momentum_x',
                'fneq_sampled_max_abs_momentum_y')):
            raise ValueError('fneq reference violates mass/momentum conservation: check convention')
        if self.ul_ref is not None and np.max(np.abs(self.ul_ref-self.u*float(self.meta['u0_lattice']))) > 2e-6:
            raise ValueError('u_lattice is not consistent with nondimensional u')
        if self.vl_ref is not None and np.max(np.abs(self.vl_ref-self.v*float(self.meta['u0_lattice']))) > 2e-6:
            raise ValueError('v_lattice is not consistent with nondimensional v')
        if len(self.x) != n or len(self.y) != n:
            raise ValueError('Coordinate length does not match dataset n')
        if self.meta.get('relaxation_mode_order') != EXPECTED_MODES:
            raise ValueError('Unexpected D2Q9 MRT moment ordering in data')
        if not np.isclose(self.meta['nu_lattice'],
                          self.meta['u0_lattice']*n/self.meta['re'], rtol=1e-5):
            raise ValueError('MRT dataset viscosity/Re metadata inconsistent')
        if not np.all(np.diff(self.t) > 0) or abs(float(self.t[0])) > 1e-8:
            raise ValueError('Time must increase from t=0 for IC sampling')
        if not test_times:
            raise ValueError('Specify at least one test time')
        test_ids = []
        for tv in test_times:
            tid = int(np.argmin(np.abs(self.t - float(tv))))
            if abs(float(self.t[tid]) - float(tv)) > 1e-5:
                raise ValueError(f'Requested test time {tv} not found in NPZ. '
                                 f'Available times: {self.t.tolist()}')
            test_ids.append(tid)
        if len(set(test_ids)) != len(test_ids):
            raise ValueError('Repeated test snapshots are not allowed')
        self.test_ids = np.array(sorted(test_ids), dtype=np.int64)
        self.train_ids = np.setdiff1d(np.arange(len(self.t)), self.test_ids)
        # Validation labels come from the reserved 10% of fixed training-time
        # macro observations, not a second block of future-time data.
        if 0 not in self.train_ids or (len(self.t)-1) not in self.train_ids:
            raise ValueError('Hold-out cannot include t=0 or t=t_end: '
                             'training must bracket every test time.')
        if not len(self.train_ids):
            raise ValueError('No training snapshots remain')
        for tid in self.test_ids:
            if not (np.any(self.train_ids < tid) and np.any(self.train_ids > tid)):
                raise ValueError(f'Test time t={self.t[tid]} is extrapolation, '
                                 'select an interior time instead')

    def _gather(self, tids, iy, ix, device, dtype):
        coords = np.column_stack([self.x[ix], self.y[iy], self.t[tids]])
        labels = np.column_stack([
            self.rho[tids, iy, ix], self.u[tids, iy, ix], self.v[tids, iy, ix]])
        # The kinetic observation is in lattice units with population order
        # c0,cE,cN,cW,cS,cNE,cNW,cSW,cSE, not physical velocity units.
        fnlabels = self.fneq_ref[tids,:,iy,ix]  # advanced indexing -> [B,9]
        if fnlabels.shape != (len(tids), 9):
            raise AssertionError('Bad kinetic observation indexing: '+str(fnlabels.shape))
        return (torch.as_tensor(coords, device=device, dtype=dtype),
                torch.as_tensor(labels, device=device, dtype=dtype),
                torch.as_tensor(fnlabels, device=device, dtype=dtype))

    def sample_random(self, ids, batch, rng, device, dtype):
        it = rng.choice(ids, size=batch, replace=True)
        iy = rng.integers(len(self.y), size=batch)
        ix = rng.integers(len(self.x), size=batch)
        return self._gather(it, iy, ix, device, dtype)

    def sample_ic(self, batch, rng, device, dtype):
        ix = rng.integers(len(self.x), size=batch)
        iy = rng.integers(len(self.y), size=batch)
        it = np.zeros(batch, dtype=np.int64)
        return self._gather(it, iy, ix, device, dtype)

    def create_observations(self, count, layout, rng):
        """Generate ONE FIXED observation set before training, never resample it.

        count=0: dense mode, sample from all training grid points each epoch.
        random-spacetime: count distinct space/time grid points.
        fixed-sensors: approximately count observations, using identical sensors
                       at every training time (actual count reported).
        """
        if count == 0:
            return None
        if count < 0:
            raise ValueError('--num-observations must be >= 0')
        n = len(self.x)
        m = len(self.train_ids)
        if layout == 'random-spacetime':
            chosen = rng.choice(m*n*n, size=min(count, m*n*n), replace=False)
            tids = self.train_ids[chosen // (n*n)]
            iy = (chosen // n) % n
            ix = chosen % n
        elif layout == 'fixed-sensors':
            n_sensors = min(n*n, max(1, round(count/m)))
            spatial = rng.choice(n*n, size=n_sensors, replace=False)
            tids = np.repeat(self.train_ids, n_sensors)
            ix = np.tile(spatial % n, m)
            iy = np.tile(spatial // n, m)
        else:
            raise ValueError('unknown observation layout')
        return {'tids': tids.astype(np.int32),
                'iy': iy.astype(np.int32), 'ix': ix.astype(np.int32)}

    def sample_observations(self, obs, batch, rng, device, dtype):
        if obs is None:
            return self.sample_random(self.train_ids, batch, rng, device, dtype)
        chosen = rng.integers(len(obs['ix']), size=batch)
        return self._gather(obs['tids'][chosen], obs['iy'][chosen], obs['ix'][chosen],
                            device, dtype)


def data_loss(pred, target, rho_scale):
    rho = torch.mean(((pred[:, 0]-target[:, 0])/rho_scale)**2)
    u = torch.mean((pred[:, 1]-target[:, 1])**2)
    v = torch.mean((pred[:, 2]-target[:, 2])**2)
    return rho+u+v, {'rho': rho, 'u': u, 'v': v}


def sample_collocation(batch, physics_end, dt, device, dtype):
    # Exclude t beyond physics_end; a streamed point adds one time step.
    if physics_end <= dt:
        raise ValueError('physics_end must exceed one lattice time step')
    xyt = torch.rand(batch, 3, device=device, dtype=dtype)
    xyt[:, 2] *= physics_end-dt
    return xyt


def fneq_observation_loss(pred, target, scale):
    # Reference fneq is O(1e-5), so unscaled MSE is ~1e-9 and would
    # be numerically dwarfed by the macro loss. Normalization is essential.
    return torch.mean(((pred-target)/scale)**2)


@torch.no_grad()
def validation_loss(model, fixed_batch, fneq_scale, lambda_fneq_obs):
    model.eval()
    coords, targets, fn_ref = fixed_batch
    pred = model.macro_state(coords)
    macro_loss, parts = data_loss(pred, targets, model.rho_scale)
    kinetic_loss = fneq_observation_loss(model.fneq(coords), fn_ref, fneq_scale)
    return (float(macro_loss + lambda_fneq_obs*kinetic_loss),
            {**{k:float(v) for k,v in parts.items()},
             'fneq':float(kinetic_loss), 'macro':float(macro_loss)})


@torch.no_grad()
def predict_snapshot(model, data, tid, device, dtype, stride=1, chunk=8192):
    xx, yy = np.meshgrid(data.x[::stride], data.y[::stride], indexing='xy')
    flat = np.column_stack([xx.ravel(), yy.ravel(),
                            np.full(xx.size, data.t[tid], dtype=np.float32)])
    model.eval()
    out = []
    for start in range(0, len(flat), chunk):
        xyt = torch.as_tensor(flat[start:start+chunk], device=device, dtype=dtype)
        out.append(model.macro_state(xyt).cpu().numpy())
    return np.concatenate(out, axis=0).reshape(len(data.y[::stride]),
                                                len(data.x[::stride]), 3)


@torch.no_grad()
def predict_fneq_snapshot(model, data, tid, device, dtype, stride=1, chunk=8192):
    xx, yy = np.meshgrid(data.x[::stride], data.y[::stride], indexing='xy')
    flat = np.column_stack([xx.ravel(), yy.ravel(),
                            np.full(xx.size, data.t[tid], dtype=np.float32)])
    model.eval()
    blocks = []
    for start in range(0, len(flat), chunk):
        xyt = torch.as_tensor(flat[start:start+chunk],device=device,dtype=dtype)
        blocks.append(model.fneq(xyt).cpu().numpy())
    return np.concatenate(blocks).reshape(len(data.y[::stride]),len(data.x[::stride]),9)


def _torch_frequency_grid(n):
    # Avoid NumPy FFT modules backed by a separate MKL/OpenMP runtime in some
    # Anaconda installations; reuse the already imported PyTorch FFT backend.
    k = torch.fft.fftfreq(n, d=1.0/n)
    try:
        ky, kx = torch.meshgrid(k, k, indexing='ij')
    except TypeError:  # old PyTorch releases
        ky, kx = torch.meshgrid(k, k)
    return kx, ky


def vorticity(u, v):
    n = u.shape[-1]
    a = torch.as_tensor(np.ascontiguousarray(u), dtype=torch.float64)
    b = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float64)
    kx, ky = _torch_frequency_grid(n)
    w = torch.fft.ifft2(1j * (2*math.pi*kx)*torch.fft.fft2(b)
                        - 1j * (2*math.pi*ky)*torch.fft.fft2(a))
    return w.real.cpu().numpy()


def rel_l2(pred, true):
    # BLAS-free: numpy.linalg.norm on MKL can load a duplicated libiomp5md.dll.
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    ref = np.asarray(true, dtype=np.float64)
    numerator = np.sqrt(np.sum(diff * diff, dtype=np.float64))
    denominator = np.sqrt(np.sum(ref * ref, dtype=np.float64))
    return float(numerator / max(denominator, 1e-30))


def energy_spectrum(u, v):
    n = u.shape[-1]
    a = torch.as_tensor(np.ascontiguousarray(u), dtype=torch.float64)
    b = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float64)
    kx, ky = _torch_frequency_grid(n)
    shell = torch.round(torch.sqrt(kx*kx+ky*ky)).long()
    modal = (torch.abs(torch.fft.fft2(a))**2
             + torch.abs(torch.fft.fft2(b))**2) / (2.0*n**4)
    max_shell = n//2
    energy = torch.bincount(shell.flatten(), weights=modal.flatten(),
                            minlength=max_shell+1)[:max_shell+1]
    return np.arange(max_shell+1), energy.cpu().numpy()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_and_save(model, data, outdir, device, dtype, stride=1, max_snapshots=0):
    ids = data.test_ids[:max_snapshots] if max_snapshots else data.test_ids
    rows, predictions, truths, errors = [], [], [], []
    vort_pred, vort_true, spec_pred, spec_true, spectra = [], [], [], [], []
    fneq_predictions, fneq_truths = [], []
    for tid in ids:
        pred = predict_snapshot(model, data, int(tid), device, dtype, stride)
        fn_pred = predict_fneq_snapshot(model,data,int(tid),device,dtype,stride)
        fn_true = np.moveaxis(data.fneq_ref[int(tid),:,::stride,::stride],0,-1)
        assert fn_pred.shape == fn_true.shape, (fn_pred.shape,fn_true.shape)
        fneq_predictions.append(fn_pred.astype(np.float32))
        fneq_truths.append(fn_true.astype(np.float32))
        true = np.stack([data.rho[tid, ::stride, ::stride],
                         data.u[tid, ::stride, ::stride],
                         data.v[tid, ::stride, ::stride]], axis=-1)
        wp, wt = vorticity(pred[..., 1], pred[..., 2]), vorticity(true[..., 1], true[..., 2])
        kvals, ep = energy_spectrum(pred[..., 1], pred[..., 2])
        _, et = energy_spectrum(true[..., 1], true[..., 2])
        rows.append({
            't': float(data.t[tid]),
            'rho_rel_l2': rel_l2(pred[..., 0], true[..., 0]),
            'fneq_rel_l2': rel_l2(fn_pred, fn_true),
            'fneq_mse_lattice': float(np.mean((fn_pred.astype(np.float64)-fn_true.astype(np.float64))**2)),
            **{f'fneq_i{j}_rel_l2':rel_l2(fn_pred[...,j],fn_true[...,j]) for j in range(9)},
            'u_rel_l2': rel_l2(pred[..., 1], true[..., 1]),
            'v_rel_l2': rel_l2(pred[..., 2], true[..., 2]),
            'vorticity_rel_l2': rel_l2(wp, wt),
            'velocity_vector_rel_l2': rel_l2(pred[..., 1:3], true[..., 1:3]),
            'energy_pred': float(np.mean(pred[..., 1]**2+pred[..., 2]**2)*0.5),
            'energy_true': float(np.mean(true[..., 1]**2+true[..., 2]**2)*0.5),
            'enstrophy_pred': float(np.mean(wp**2)*0.5),
            'enstrophy_true': float(np.mean(wt**2)*0.5),
        })
        predictions.append(pred.astype(np.float32))
        truths.append(true.astype(np.float32))
        errors.append(abs(pred-true).astype(np.float32))
        vort_pred.append(wp.astype(np.float32))
        vort_true.append(wt.astype(np.float32))
        spec_pred.append(ep)
        spec_true.append(et)
        for k, e0, e1 in zip(kvals, et, ep):
            spectra.append({'t': float(data.t[tid]), 'k': int(k),
                            'energy_true': float(e0), 'energy_pred': float(e1)})
    write_csv(outdir/'test_metrics.csv', rows)
    write_csv(outdir/'energy_spectra.csv', spectra)
    (outdir/'test_metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    np.savez_compressed(outdir/'test_fields.npz',
                        time=data.t[ids], x=data.x[::stride], y=data.y[::stride],
                        prediction=np.stack(predictions), truth=np.stack(truths),
                        abs_error=np.stack(errors),
                        vorticity_prediction=np.stack(vort_pred),
                        vorticity_truth=np.stack(vort_true),
                        spectrum_k=kvals, spectrum_prediction=np.stack(spec_pred),
                        spectrum_truth=np.stack(spec_true),
                        fneq_prediction=np.stack(fneq_predictions),
                        fneq_truth=np.stack(fneq_truths))
    return rows


def save_checkpoint(path, model, optimizer, scheduler, phase, epoch,
                    history, args, data, best_scores, rng):
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'phase': phase, 'epoch': epoch,
        'history': history, 'args': vars(args), 'metadata': data.meta,
        'best_scores': best_scores, 'numpy_rng_state': rng.bit_generator.state,
        'torch_rng_state': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    torch.save(state, path)



def load_checkpoint_compat(path, device):
    """Read a checkpoint made by this script across older and newer PyTorch.

    IMPORTANT: full PyTorch checkpoints contain pickled objects. Only load
    checkpoints you generated yourself or otherwise trust.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        # PyTorch versions that do not expose the weights_only keyword.
        return torch.load(path, map_location=device)


def restore_saved_experiment_args(args):
    """Restore all model/training/split settings for post-crash evaluation.

    The current CLI still controls the output folder and evaluation resolution.
    No optimizer steps are performed in --evaluate-only mode.
    """
    config_path = args.output / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(
            'Cannot evaluate the previous run: missing ' + str(config_path))
    saved = json.loads(config_path.read_text(encoding='utf-8'))
    if saved.get('design_id') != 'fulltime_holdout_inverse_mrt_fneq_v1':
        raise ValueError('Checkpoint belongs to another data split; cannot load '
                         'it for full-time hold-out testing. Use the ORIGINAL script '
                         'to evaluate the old model, or train a new full-time model.')
    cli_eval_stride = args.eval_stride
    cli_eval_max = args.eval_max_snapshots
    cli_skip_plots = args.skip_plots
    cli_plot_dpi = args.plot_dpi
    cli_data = args.data
    for key in vars(args):
        if key in saved and key not in ('output', 'evaluate_only', 'resume', 'smoke'):
            value = saved[key]
            setattr(args, key, Path(value) if key == 'data' else value)
    # Evaluation CLI always controls resolution and number of tested snapshots.
    # Default eval_stride=1 means FULL resolution, even if prior --smoke saved 8.
    args.eval_stride = cli_eval_stride
    args.eval_max_snapshots = cli_eval_max
    args.skip_plots = cli_skip_plots
    args.plot_dpi = cli_plot_dpi
    args.smoke = False
    args.resume = None
    if not args.data.exists():
        # Helpful when the previous experiment directory moved with the data.
        candidates = [Path(__file__).resolve().parent / args.data.name, cli_data]
        args.data = next((c for c in candidates if c.exists()), args.data)
    if not args.data.exists():
        raise FileNotFoundError('Dataset for stored checkpoint not found: '+str(args.data))
    print('[EVALUATION ONLY] Loaded settings from:', config_path)
    print('[EVALUATION ONLY] No training will be repeated.')
    return args

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent /
                   'dsl_Re1000_N256_MaSafe_with_fneq.npz')
    p.add_argument('--output', type=Path, default=Path('results/double_kan_mrt_inverse_fneq_MRT_only'))
    p.add_argument('--width', type=int, default=48)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--grid-size', type=int, default=8)
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--warmup-epochs', type=int, default=1500)
    p.add_argument('--physics-epochs', type=int, default=3500)
    p.add_argument('--data-batch', type=int, default=2048)
    p.add_argument('--physics-batch', type=int, default=512)
    p.add_argument('--ic-batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lambda-physics', type=float, default=0.05)
    p.add_argument('--lambda-ic', type=float, default=1.0)
    p.add_argument('--lambda-positive', type=float, default=0.01)
    p.add_argument('--lambda-fneq', type=float, default=0.0,
                   help='Optional size regularization of fneq (usually OFF with supervised reference)')
    p.add_argument('--lambda-fneq-obs', type=float, default=1.0,
                   help='Weight on direct ground-truth fneq supervision at all macro observation positions')
    p.add_argument('--lambda-ic-fneq', type=float, default=0.2,
                   help='Weight on reference pre-collision fneq at t=0')
    p.add_argument('--fneq-obs-norm', type=float, default=1e-4,
                   help='Scale for kinetic supervision in lattice population units')
    p.add_argument('--fneq-scale', type=float, default=0.0002,
                   help='Kinetic branch output scale in lattice distribution units')
    p.add_argument('--num-observations', type=int, default=10000,
                   help='Budget of distinct macro observations over all NON-TEST times, split 90/10 train/validation')
    p.add_argument('--observation-layout', choices=['random-spacetime', 'fixed-sensors'],
                   default='random-spacetime')
    p.add_argument('--test-times', nargs='+', type=float, default=[0.5, 1.0, 1.5, 1.9],
                   help='Complete temporal snapshots reserved EXCLUSIVELY for evaluation; '
                        'each must be interior to the training time coverage')
    p.add_argument('--validation-samples', type=int, default=4096)
    p.add_argument('--validation-fraction', type=float, default=0.1,
                   help='Fraction of fixed labelled observations reserved for validation')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--learn-nu', dest='learn_nu', action='store_true',
                      help='Inverse identification: optimize a positive viscosity (default)')
    mode.add_argument('--fixed-nu', dest='learn_nu', action='store_false',
                      help='Reconstruction: fix viscosity to the reference dataset value')
    p.set_defaults(learn_nu=True)
    p.add_argument('--nu-init', type=float, default=0.002,
                   help='Initial nondimensional viscosity for inverse mode (NOT true nu)')
    p.add_argument('--skip-plots', action='store_true',
                   help='Skip automatic PNG generation after test evaluation')
    p.add_argument('--plot-dpi', type=int, default=300,
                   help='Resolution of flow field images (PNG)')
    p.add_argument('--evaluate-only', action='store_true',
                   help='Recover a previous finished run without retraining; reads '
                        'the original settings from OUTPUT/config.json')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--print-every', type=int, default=50)
    p.add_argument('--eval-stride', type=int, default=1,
                   help='1=full-resolution evaluation; larger for quick diagnostics only')
    p.add_argument('--eval-max-snapshots', type=int, default=0,
                   help='0=all test snapshots; other values for diagnostics only')
    p.add_argument('--threads', type=int, default=0,
                   help='Set PyTorch CPU threads (0=PyTorch default)')
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--smoke', action='store_true', help='Tiny run, NOT a convergence test')
    return p.parse_args()


def main():
    args = parse_args()
    if args.evaluate_only:
        args = restore_saved_experiment_args(args)
    if args.smoke:
        args.width, args.depth, args.grid_size = 12, 1, 4
        args.warmup_epochs, args.physics_epochs = 2, 2
        args.data_batch, args.ic_batch, args.physics_batch = 64, 64, 16
        args.num_observations = 128
        args.validation_samples = 128
        args.print_every, args.eval_stride, args.eval_max_snapshots = 1, 8, 1
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.learn_nu and (args.nu_init is None or args.nu_init <= 0):
        raise ValueError('inverse mode requires --nu-init > 0')
    if args.lambda_fneq_obs < 0 or args.fneq_obs_norm <= 0:
        raise ValueError('Kinetic supervision weight must be nonnegative and normalization must be positive')
    if args.lambda_ic_fneq < 0:
        raise ValueError('IC kinetic supervision weight must be nonnegative')
    if args.learn_nu and args.physics_epochs == 0:
        raise ValueError('inverse mode requires physics training epochs > 0')
    if args.eval_stride < 1 or args.print_every < 1:
        raise ValueError('eval-stride and print-every must be >=1')
    if not (0 < args.validation_fraction < 1):
        raise ValueError('validation-fraction must be between 0 and 1')
    if args.num_observations < 2:
        raise ValueError('--num-observations >= 2 is required for fixed 90/10 validation')
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64 if args.precision == 'float64' else torch.float32
    data = TrajectoryData(args.data, args.test_times)
    meta = data.meta
    print('[DATA QUALITY]',json.dumps(data.quality_report,ensure_ascii=False))
    rates = meta['relaxation_rates']
    model = DualKANMRT(
        n=meta['n'], u0=meta['u0_lattice'], nu_lattice=meta['nu_lattice'],
        base_rates=rates, t_end=meta['t_end'], width=args.width,
        depth=args.depth, grid_size=args.grid_size, learn_nu=args.learn_nu,
        nu_init=args.nu_init, fneq_scale=args.fneq_scale).to(device=device, dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.warmup_epochs+args.physics_epochs), eta_min=args.lr/20)
    rng = np.random.default_rng(args.seed)
    obs = data.create_observations(args.num_observations, args.observation_layout, rng)
    total_fixed_observations = len(obs['ix']) if obs is not None else None
    vrng = np.random.default_rng(args.seed+12345)
    # Reserve a fixed validation subset from training-time observations.
    # All ground-truth labels at the selected TEST TIMES remain completely unused.
    if obs is None or len(obs['ix']) < 2:
        raise ValueError('This experiment requires at least two fixed observations')
    eligible = np.flatnonzero(obs['tids'] != 0)
    if len(eligible) < 1:
        raise ValueError('Need observations after t=0 for validation')
    val_select = vrng.permutation(eligible)[:max(1, min(len(eligible),
                                             round(len(obs['ix'])*args.validation_fraction)))]
    keep = np.ones(len(obs['ix']), dtype=bool)
    keep[val_select] = False
    train_select = np.flatnonzero(keep)
    validation_obs = {k: v[val_select] for k, v in obs.items()}
    obs = {k: v[train_select] for k, v in obs.items()}
    if np.isin(obs['tids'], data.test_ids).any() or np.isin(validation_obs['tids'], data.test_ids).any():
        raise AssertionError('Test ground-truth data leaked into training or validation')
    # FULL-TIME physics: sample collocation in [0, t_end-dt], including the
    # neighborhoods of held-out test times (without test velocity labels).
    physics_end = float(meta['t_end'])
    dt = float(meta['u0_lattice'])/int(meta['n'])
    if physics_end <= dt:
        raise ValueError('Invalid MRT lattice time step')
    if abs(float(data.t[-1])-physics_end) > 1e-5:
        raise ValueError('Dataset last timestamp must equal metadata t_end')
    if not args.evaluate_only and not args.resume and (args.output/'config.json').exists():
        raise FileExistsError('Output directory already contains an experiment. '
                              'Use a NEW --output path or --resume to continue the matching run.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'data_quality_report.json').write_text(
        json.dumps(data.quality_report,indent=2),encoding='utf-8')
    if obs is not None and not args.evaluate_only:
        np.savez_compressed(args.output/'observation_indices.npz', **obs)
    if not args.evaluate_only:
        np.savez_compressed(args.output/'validation_observation_indices.npz', **validation_obs)
    # Validation uses DISTINCT held-out labels, without replacement.
    vtake = min(args.validation_samples, len(validation_obs['ix']))
    vsel = vrng.choice(len(validation_obs['ix']), size=vtake, replace=False)
    val_batch = data._gather(validation_obs['tids'][vsel],
                             validation_obs['iy'][vsel],
                             validation_obs['ix'][vsel], device, dtype)
    print('EXPERIMENT MODE:', 'INVERSE (nu trainable)' if args.learn_nu else 'RECONSTRUCTION (nu fixed)')
    print(f'device={device}, dtype={dtype}, parameters={sum(p.numel() for p in model.parameters()):,}')
    print(f'NPZ: n={meta["n"]}, U0={meta["u0_lattice"]}, nu_lat(true)={meta["nu_lattice"]}, '
          f'dt={dt:.8g}; physics_end={physics_end}')
    print(f'FULL TIME: labeled train time range [{float(data.t[data.train_ids].min()):.3f}, '
          f'{float(data.t[data.train_ids].max()):.3f}], '
          f'{len(data.train_ids)} training snapshots')
    print(f'TEST snapshots (NO labeled observations): {data.t[data.test_ids].tolist()}')
    print(f'PHYSICS collocation time: [0, {physics_end-dt:.8f}] '
          f'(streaming reaches t={physics_end})')
    print(f'fixed observations (total / train / validation) = '
          f'{total_fixed_observations} / '
          f'{len(obs["ix"]) if obs is not None else "dense-online"} / '
          f'{len(validation_obs["ix"]) if validation_obs is not None else "future times"}; '
          f'layout={args.observation_layout}')
    print(f'PHYSICS: discrete D2Q9 MRT only; weight={args.lambda_physics}; '
          f'kinetic fneq observation weight={args.lambda_fneq_obs}')
    print(f'[KINETIC SUPERVISION] pre-collision fneq on same {len(obs["ix"])} macro observation points; weight={args.lambda_fneq_obs}, normalized by {args.fneq_obs_norm:g}')
    if args.learn_nu:
        print(f'inverse initial nondimensional nu={args.nu_init}; reference nu hidden from training')

    best_scores = {'warmup': float('inf'), 'physics': float('inf')}
    history, start_phase, start_epoch = [], 'warmup', 0
    if args.resume:
        ckpt = load_checkpoint_compat(args.resume, device)
        if 'lambda_ns' in ckpt.get('args', {}):
            raise ValueError('Cannot resume a previous MRT+NS experiment into MRT-only training.')
        for name in ('lambda_physics', 'lambda_fneq_obs', 'lambda_ic_fneq',
                     'fneq_obs_norm', 'fneq_scale', 'num_observations', 'precision',
                     'width', 'depth', 'grid_size', 'seed', 'nu_init', 'learn_nu'):
            if ckpt['args'].get(name) != getattr(args, name):
                raise ValueError(f'Cannot resume: {name} differs from saved checkpoint')
        if bool(ckpt['args']['learn_nu']) != args.learn_nu:
            raise ValueError('resume learn-nu option does not match checkpoint')
        if ckpt['args'].get('test_times') != args.test_times:
            raise ValueError('Checkpoint was trained with different time split or '
                             'old forecast code: cannot resume into this experiment.')
        if int(ckpt['metadata']['n']) != int(meta['n']):
            raise ValueError('Checkpoint dataset grid does not match')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        history = ckpt['history']
        best_scores = ckpt.get('best_scores', best_scores)
        rng.bit_generator.state = ckpt['numpy_rng_state']
        torch.random.set_rng_state(ckpt['torch_rng_state'].cpu())
        if torch.cuda.is_available() and 'cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        start_phase, start_epoch = ckpt['phase'], ckpt['epoch']+1
        print(f'resuming from {args.resume} after {start_phase} epoch={start_epoch-1}')

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(design_id='fulltime_holdout_inverse_mrt_fneq_v1', metadata=meta,
                  physics_end=physics_end, discrete_time_step=dt,
                  number_of_fixed_observations=(total_fixed_observations or 0),
                  number_of_training_observations=(len(obs['ix']) if obs is not None else 0),
                  number_of_validation_observations=(len(validation_obs['ix'])
                                                     if validation_obs is not None else 0),
                  train_indices=data.train_ids.tolist(),
                  validation_indices_source='held-out fixed training-time observations',
                  test_indices=data.test_ids.tolist())
    if not args.evaluate_only:
        (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    tic = time.time()
    if args.evaluate_only:
        # The old script crashed after writing checkpoints, before testing.
        # Recover the saved states; never optimize or overwrite checkpoints.
        last_phase = 'physics' if args.physics_epochs else 'warmup'
        final_path = args.output/'final.pt'
        if final_path.is_file():
            final_ckpt = load_checkpoint_compat(final_path, device)
            model.load_state_dict(final_ckpt['model'])
            final_nu = float(model.nu_nondim.detach())
            best_scores = final_ckpt.get('best_scores', best_scores)
        else:
            final_nu = float('nan')
            print('WARNING: final.pt missing; only best checkpoint can be reported')
    else:
        for phase, epochs in [('warmup', args.warmup_epochs), ('physics', args.physics_epochs)]:
            if start_phase == 'physics' and phase == 'warmup':
                continue
            initial = start_epoch if phase == start_phase else 0
            for epoch in range(initial, epochs):
                model.train()
                opt.zero_grad(set_to_none=True)
                xyt_d, target, target_fneq = data.sample_observations(obs, args.data_batch, rng, device, dtype)
                l_data, parts = data_loss(model.macro_state(xyt_d), target, model.rho_scale)
                l_fneq_obs = fneq_observation_loss(model.fneq(xyt_d),target_fneq,args.fneq_obs_norm)
                xyt_ic, target_ic, target_fneq_ic = data.sample_ic(args.ic_batch, rng, device, dtype)
                l_ic, _ = data_loss(model.macro_state(xyt_ic), target_ic, model.rho_scale)
                l_fneq_ic = fneq_observation_loss(model.fneq(xyt_ic),target_fneq_ic,args.fneq_obs_norm)
                zero = l_data.new_zeros(())
                l_phys, l_pos, l_neq = zero, zero, zero
                if phase == 'physics':
                    xyt_f = sample_collocation(args.physics_batch, physics_end, dt, device, dtype)
                    terms = model.physics_terms(xyt_f)
                    # Multiplication by N converts the one-step residual to an O(1)
                    # nondimensional transport/collision residual.
                    l_phys = (model.n*terms['residual']).square().mean()
                    l_pos = terms['positivity'].square().mean()
                    l_neq = terms['fneq'].square().mean()
                loss = (l_data + args.lambda_ic*l_ic
                        + args.lambda_fneq_obs*l_fneq_obs
                        + args.lambda_ic_fneq*l_fneq_ic
                        + args.lambda_physics*l_phys
                        + args.lambda_positive*l_pos + args.lambda_fneq*l_neq)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite loss in {phase} epoch={epoch}')
                loss.backward()
                log_nu_grad = (float(model.log_nu.grad.detach().cpu())
                               if args.learn_nu and model.log_nu.grad is not None else float('nan'))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
                scheduler.step()
                def val(z):
                    return float(z.detach().cpu())
                rec = {
                    'phase': phase, 'epoch': epoch, 'global_epoch': len(history),
                    'total_loss': val(loss), 'data_loss': val(l_data), 'ic_loss': val(l_ic),
                    'fneq_observation_loss': val(l_fneq_obs), 'fneq_ic_observation_loss': val(l_fneq_ic),
                    'mrt_loss_scaled': val(l_phys),
                    'positivity_loss': val(l_pos), 'fneq_loss': val(l_neq),
                    'nu_log_grad_pre_clip': log_nu_grad,
                    'rho_mse_scaled': val(parts['rho']),
                    'u_mse': val(parts['u']), 'v_mse': val(parts['v']),
                    'nu_nondim': val(model.nu_nondim),
                    'nu_lattice': val(model.nu_lattice), 'lr': opt.param_groups[0]['lr'],
                }
                history.append(rec)
                if epoch % args.print_every == 0 or epoch+1 == epochs:
                    vloss, vparts = validation_loss(model, val_batch, args.fneq_obs_norm, args.lambda_fneq_obs)
                    rec.update(validation_loss=vloss, validation_rho=vparts['rho'],
                               validation_u=vparts['u'], validation_v=vparts['v'],
                               validation_fneq=vparts['fneq'],validation_macro=vparts['macro'])
                    print(f'{phase:7s} {epoch+1:5d}/{epochs} total={val(loss):.3e} '
                          f'data={val(l_data):.3e} ic={val(l_ic):.3e} '
                          f'fneq={val(l_fneq_obs):.3e} MRT={val(l_phys):.3e} val={vloss:.3e} '
                          f'nu={val(model.nu_nondim):.7g}')
                    if vloss < best_scores[phase]:
                        best_scores[phase] = vloss
                        save_checkpoint(args.output/f'best_{phase}.pt', model, opt, scheduler,
                                        phase, epoch, history, args, data, best_scores, rng)
                    write_csv(args.output/'history.csv', history)
                    (args.output/'history.json').write_text(
                        json.dumps(history, indent=2), encoding='utf-8')
            start_epoch = 0
            if epochs:
                save_checkpoint(args.output/f'after_{phase}.pt', model, opt, scheduler,
                                phase, epochs-1, history, args, data, best_scores, rng)

        last_phase = 'physics' if args.physics_epochs else 'warmup'
        save_checkpoint(args.output/'final.pt', model, opt, scheduler,
                        last_phase, max(0, args.physics_epochs-1 if args.physics_epochs else args.warmup_epochs-1),
                        history, args, data, best_scores, rng)
        final_nu = float(model.nu_nondim.detach())
    # Critically: never select a warmup-only best model for a physics/inverse run.
    selected_path = args.output/f'best_{last_phase}.pt'
    if not selected_path.exists():
        raise RuntimeError(f'best checkpoint missing: {selected_path}')
    best = load_checkpoint_compat(selected_path, device)
    model.load_state_dict(best['model'])
    best_scores = best.get('best_scores', best_scores)
    best_nu = float(model.nu_nondim.detach())
    rows = evaluate_and_save(model, data, args.output, device, dtype,
                             stride=args.eval_stride, max_snapshots=args.eval_max_snapshots)
    reference_nu = float(meta['nu_lattice'])/(meta['u0_lattice']*meta['n'])
    report = {
        'experiment_mode': 'inverse' if args.learn_nu else 'fixed_viscosity_reconstruction',
        'viscosity_was_optimized': bool(args.learn_nu),
        'evaluation_only_recovery': bool(args.evaluate_only),
        'training_method': 'global full-time macro AND pre-collision fneq supervision (excluding held-out test snapshots)',
        'selected_checkpoint': str(selected_path),
        'best_validation_loss': best_scores[last_phase],
        'selected_nu_nondim': best_nu,
        'final_nu_nondim': final_nu,
        'reference_nu_nondim_FOR_REPORT_ONLY': reference_nu,
        'viscosity_relative_error_if_inverse': (abs(best_nu-reference_nu)/reference_nu
                                                 if args.learn_nu else None),
        'viscosity_note': ('The viscosity was optimized, but accuracy must be checked.'
                           if args.learn_nu else
                           'The viscosity was fixed to the reference; this is NOT inversion.'),
        'observation_count_total': total_fixed_observations,
        'observation_count_training': len(obs['ix']) if obs is not None else None,
        'observation_count_validation': (len(validation_obs['ix'])
                                         if validation_obs is not None else None),
        'checkpoint_selection_uses_extra_time_snapshots': False,
        'physics_interval': [0.0, physics_end],
        'physics_equation': 'discrete D2Q9 MRT-LBM collision/streaming only',
        'kinetic_data_supervision': 'pre-collision reference fneq, 9 components, same samples as macro',
        'fneq_observation_weight':args.lambda_fneq_obs,
        'fneq_observation_normalization':args.fneq_obs_norm,
        'fneq_ic_weight':args.lambda_ic_fneq,
        'data_quality_report':data.quality_report,
        'test_is_temporal_extrapolation': False,
        'training_time_range': [float(data.t[data.train_ids].min()), float(data.t[data.train_ids].max())],
        'test_times': [float(t) for t in data.t[data.test_ids]],
        'test_time_labels_excluded_from_training': True,
        'test_fneq_labels_excluded_from_training': True,
        'physics_collocation_covers_test_time_interval': True,
        'validation_strategy': '10 percent held-out fixed macro+fneq observations from training snapshots',
        'evaluation_downsampled': args.eval_stride != 1,
        'evaluated_snapshots': len(rows),
        'elapsed_seconds': time.time()-tic,
        'warning': ('Reference pre-collision fneq is supervised at labeled times. '
                    'Neither direct fneq supervision nor MRT residual alone guarantees '
                    'unique or accurate viscosity identification.'),
    }
    (args.output/'run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output/'source_data_manifest.json').write_text(json.dumps({
        'history.csv': 'One record per optimization epoch.',
        'observation_indices.npz': 'Fixed TRAIN macro+fneq observation indices, when used.',
        'data_quality_report.json': 'MRT population conservation checks; NOT a convergence guarantee.',
        'validation_observation_indices.npz': 'Held-out validation indices from fixed observation budget.',
        'test_snapshot_indices.json': 'Complete test snapshot indices, NEVER used in macro supervised training.',
        'best_warmup.pt': 'Best macro-only pretraining checkpoint.',
        'best_physics.pt': 'Best joint MRT-constrained checkpoint, used for test.',
        'test_metrics.csv': 'Held-out macro, vorticity and 9-population fneq errors and physical integrals.',
        'energy_spectra.csv': 'Integer-shell 2D FFT kinetic energy.',
        'test_fields.npz': 'Held-out reference/predictions, macro and kinetic fields and spectra.',
        'run_summary.json': 'Selected checkpoint and viscosity result.',
        'limitation': 'Reference pre-collision fneq directly supervised. '
                      'No macroscopic momentum or pressure loss. No guarantee of viscosity identifiability.'
    }, indent=2), encoding='utf-8')
    (args.output/'test_snapshot_indices.json').write_text(json.dumps({
        'test_indices': data.test_ids.tolist(),
        'test_times': [float(t) for t in data.t[data.test_ids]],
        'training_indices': data.train_ids.tolist(),
        'no_supervised_test_labels': True,
        'no_test_fneq_labels_in_training_or_validation': True,
        'MRT_physics_collocation_spans_test_times': True,
    }, indent=2), encoding='utf-8')
    print('selected checkpoint:', selected_path)
    print(json.dumps(report, indent=2))
    print('test metrics:', json.dumps(rows, indent=2))

    if not args.skip_plots:
        # Use THIS SAME FILE as the plotter in a separate Python process.
        # --plot-only exits before importing torch, reducing Windows OpenMP
        # duplicate runtime conflicts while requiring only one downloaded file.
        cmd = [sys.executable, str(Path(__file__).resolve()),
               '--plot-only', '--output', str(args.output),
               '--dpi', str(args.plot_dpi)]
        print('[PLOTS] Auto-generating flow, error, vorticity, spectra '
              'and training figures...', flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f'WARNING: plotting subprocess exited {result.returncode}. '
                  f'Numeric results are still saved. You can retry without training: '
                  f'{Path(__file__).name} --plot-only --output {args.output}')
        else:
            print('[PLOTS] All PNG files saved under:', args.output/'figures')

if __name__ == '__main__':
    main()
